//! Native backend: SQLite on a dedicated worker thread.

use crate::db::DbConfig;
use crate::error::{DbError, Result};
use crate::migrate::{Migration, MigrationReport};
use crate::value::{Row, Rows, Value};
use futures::channel::oneshot;
use rusqlite::types::{ToSql, ToSqlOutput, ValueRef};
use rusqlite::{Connection, params_from_iter};
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc;

type Job = Box<dyn FnOnce(&mut std::result::Result<Connection, String>) + Send>;

/// Owns the sending side of the job queue; the thread owns the connection.
pub(crate) struct Worker {
    tx: mpsc::Sender<Job>,
    /// Jobs submitted and not finished yet.
    depth: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Worker {
    pub(crate) fn spawn(config: &DbConfig) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let config = config.clone();
        std::thread::Builder::new()
            .name("space3d-db".into())
            .spawn(move || {
                let mut conn = open(&config).map_err(|e| {
                    tracing::error!(path = %config.path.display(), error = %e, "failed to open database");
                    e.to_string()
                });
                while let Ok(job) = rx.recv() {
                    job(&mut conn);
                }
                tracing::debug!(path = %config.path.display(), "db worker shutting down");
            })
            .expect("failed to spawn space3d-db worker thread");
        Self {
            tx,
            depth: Default::default(),
        }
    }

    /// Jobs queued or running.
    pub(crate) fn depth(&self) -> usize {
        self.depth.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Queue `f` on the worker. Panics inside `f` are caught and reported as
    /// [`DbError::Panicked`]; the worker keeps serving later jobs.
    pub(crate) fn submit<R, F>(&self, f: F) -> oneshot::Receiver<Result<R>>
    where
        R: Send + 'static,
        F: FnOnce(&mut Connection) -> Result<R> + Send + 'static,
    {
        let (otx, orx) = oneshot::channel();
        let depth = self.depth.clone();
        depth.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let job: Job = Box::new(move |conn| {
            let done = DepthGuard(depth);
            let result = match conn {
                Ok(conn) => catch_unwind(AssertUnwindSafe(|| f(conn))).unwrap_or_else(|p| {
                    let msg = p
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| p.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "unknown panic".into());
                    Err(DbError::Panicked(msg))
                }),
                Err(msg) => Err(DbError::Open(msg.clone())),
            };
            // Count the job as finished before its result is observable: whoever sees the result
            // must also see the queue depth without this job.
            drop(done);
            let _ = otx.send(result);
        });
        // If the worker is gone the job (and `otx`) is dropped: the receiver reports WorkerGone.
        let _ = self.tx.send(job);
        orx
    }
}

/// Decrements the queue depth when a job finishes (or is dropped unrun).
struct DepthGuard(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl Drop for DepthGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Tables backing the built-in stores. `space3d_*` names are reserved.
const BUILTIN_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS space3d_kv (
    namespace  TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (namespace, key)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS space3d_saves (
    slot       TEXT PRIMARY KEY NOT NULL,
    label      TEXT NOT NULL,
    metadata   TEXT NOT NULL,
    data       BLOB NOT NULL,
    thumbnail  BLOB,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    seq        INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS space3d_leaderboard (
    board       TEXT NOT NULL,
    player      TEXT NOT NULL,
    score       INTEGER NOT NULL,
    achieved_at INTEGER NOT NULL,
    seq         INTEGER NOT NULL,
    PRIMARY KEY (board, player)
);
CREATE INDEX IF NOT EXISTS space3d_leaderboard_by_score ON space3d_leaderboard(board, score);
";

fn open(config: &DbConfig) -> Result<Connection> {
    let conn = if config.is_in_memory() {
        Connection::open_in_memory()?
    } else {
        if let Some(parent) = config.path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| DbError::Open(format!("{}: {e}", parent.display())))?;
        }
        Connection::open(&config.path)?
    };
    conn.set_prepared_statement_cache_capacity(config.statement_cache_capacity);
    conn.busy_timeout(config.busy_timeout)?;
    if config.wal && !config.is_in_memory() {
        let mode: String = conn.query_row("PRAGMA journal_mode = WAL", [], |r| r.get(0))?;
        if !mode.eq_ignore_ascii_case("wal") {
            tracing::warn!(mode, "could not enable WAL journal mode");
        }
    }
    conn.execute_batch(
        "PRAGMA synchronous = NORMAL;
         PRAGMA foreign_keys = ON;
         PRAGMA temp_store = MEMORY;",
    )?;
    conn.execute_batch(BUILTIN_SCHEMA)?;
    Ok(conn)
}

impl ToSql for Value {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::Borrowed(match self {
            Value::Null => ValueRef::Null,
            Value::Integer(i) => ValueRef::Integer(*i),
            Value::Real(f) => ValueRef::Real(*f),
            Value::Text(s) => ValueRef::Text(s.as_bytes()),
            Value::Blob(b) => ValueRef::Blob(b),
        }))
    }
}

impl From<ValueRef<'_>> for Value {
    fn from(v: ValueRef<'_>) -> Self {
        match v {
            ValueRef::Null => Value::Null,
            ValueRef::Integer(i) => Value::Integer(i),
            ValueRef::Real(f) => Value::Real(f),
            ValueRef::Text(t) => Value::Text(String::from_utf8_lossy(t).into_owned()),
            ValueRef::Blob(b) => Value::Blob(b.to_vec()),
        }
    }
}

pub(crate) fn execute(conn: &Connection, sql: &str, params: &[Value]) -> Result<usize> {
    Ok(conn.prepare_cached(sql)?.execute(params_from_iter(params))?)
}

/// Run a cached statement and hand each row (as borrowed values) to `f`.
pub(crate) fn query_with<T>(
    conn: &Connection,
    sql: &str,
    params: &[Value],
    limit: Option<usize>,
    mut f: impl FnMut(Row<'_>) -> Result<T>,
) -> Result<Vec<T>> {
    let mut stmt = conn.prepare_cached(sql)?;
    let columns: Vec<String> = stmt.column_names().into_iter().map(String::from).collect();
    let mut rows = stmt.query(params_from_iter(params))?;
    let mut out = Vec::new();
    let mut buf = Vec::with_capacity(columns.len());
    while limit.is_none_or(|l| out.len() < l) {
        let Some(row) = rows.next()? else { break };
        buf.clear();
        for i in 0..columns.len() {
            buf.push(Value::from(row.get_ref(i)?));
        }
        out.push(f(Row::new(&columns, &buf))?);
    }
    Ok(out)
}

pub(crate) fn query_raw(conn: &Connection, sql: &str, params: &[Value]) -> Result<Rows> {
    let mut stmt = conn.prepare_cached(sql)?;
    let columns: Vec<String> = stmt.column_names().into_iter().map(String::from).collect();
    let n = columns.len();
    let mut rows = stmt.query(params_from_iter(params))?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(
            (0..n)
                .map(|i| row.get_ref(i).map(Value::from))
                .collect::<rusqlite::Result<Vec<_>>>()?,
        );
    }
    Ok(Rows { columns, rows: out })
}

pub(crate) fn apply_migrations(conn: &mut Connection, migrations: &[Migration]) -> Result<MigrationReport> {
    let mut sorted: Vec<&Migration> = migrations.iter().collect();
    sorted.sort_by_key(|m| m.version);
    if let Some(w) = sorted.windows(2).find(|w| w[0].version == w[1].version) {
        return Err(DbError::Migration {
            version: w[1].version,
            name: w[1].name.to_string(),
            message: format!("duplicate migration version (also used by `{}`)", w[0].name),
        });
    }
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_migrations (
            version    INTEGER PRIMARY KEY NOT NULL,
            name       TEXT NOT NULL,
            applied_at INTEGER NOT NULL
        )",
    )?;
    let applied: HashMap<i64, String> = {
        let mut stmt = conn.prepare("SELECT version, name FROM schema_migrations")?;
        stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?
    };
    let mut report = MigrationReport::default();
    for m in sorted {
        if let Some(name) = applied.get(&m.version) {
            if name != &m.name {
                tracing::warn!(
                    version = m.version,
                    applied = %name,
                    declared = %m.name,
                    "migration name differs from the one recorded in schema_migrations"
                );
            }
            continue;
        }
        let fail = |e: rusqlite::Error| DbError::Migration {
            version: m.version,
            name: m.name.to_string(),
            message: e.to_string(),
        };
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(fail)?;
        tx.execute_batch(&m.up_sql).map_err(fail)?;
        tx.execute(
            "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?1, ?2, ?3)",
            rusqlite::params![m.version, m.name.as_ref(), crate::now_ms()],
        )
        .map_err(fail)?;
        tx.commit().map_err(fail)?;
        tracing::info!(version = m.version, name = %m.name, "applied migration");
        report.applied.push(m.version);
    }
    report.current_version = conn.query_row("SELECT MAX(version) FROM schema_migrations", [], |r| r.get(0))?;
    Ok(report)
}
