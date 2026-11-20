//! Non-blocking HTTP client ([`Http`] resource).
//!
//! Every call returns a [`Task`] immediately; the request runs on the
//! background runtime (reqwest on tokio natively, reqwest's `fetch` backend on
//! the web). Poll the task from a system, or use the `*_then` variants to get a
//! main-thread callback with full [`SystemCtx`] access once it completes.

use crate::backoff::RetryPolicy;
use crate::error::error_chain;
use crate::{NetError, rt};
use parking_lot::Mutex;
use serde::Serialize;
use serde::de::DeserializeOwned;
use space3d_core::SystemCtx;
use space3d_tasks::{MaybeSend, Task, TaskState, spawn};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

pub use http::Method;

/// An HTTP request description. Cheap to build; nothing happens until it is
/// passed to [`Http::request`].
#[derive(Clone, Debug, PartialEq)]
pub struct HttpRequest {
    /// HTTP method.
    pub method: Method,
    /// Absolute URL, or relative to [`HttpConfig::base_url`] (on the web: to the page URL).
    pub url: String,
    /// Extra headers; override [`HttpConfig::headers`] with the same name.
    pub headers: Vec<(String, String)>,
    /// Request body (empty = none).
    pub body: Vec<u8>,
    /// Per-request timeout, overriding [`HttpConfig::timeout`].
    pub timeout: Option<Duration>,
    /// Per-request retry policy, overriding [`HttpConfig::retry`].
    pub retry: Option<RetryPolicy>,
}

impl HttpRequest {
    /// Request with an arbitrary method.
    pub fn new(method: Method, url: impl Into<String>) -> Self {
        Self {
            method,
            url: url.into(),
            headers: Vec::new(),
            body: Vec::new(),
            timeout: None,
            retry: None,
        }
    }
    /// `GET url`.
    pub fn get(url: impl Into<String>) -> Self {
        Self::new(Method::GET, url)
    }
    /// `POST url` (add a body with [`Self::body`] / [`Self::json`]).
    pub fn post(url: impl Into<String>) -> Self {
        Self::new(Method::POST, url)
    }
    /// `PUT url`.
    pub fn put(url: impl Into<String>) -> Self {
        Self::new(Method::PUT, url)
    }
    /// `DELETE url`.
    pub fn delete(url: impl Into<String>) -> Self {
        Self::new(Method::DELETE, url)
    }
    /// Add a header.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
    /// Set a raw body.
    pub fn body(mut self, body: impl Into<Vec<u8>>) -> Self {
        self.body = body.into();
        self
    }
    /// Serialize `value` as the JSON body and set `content-type: application/json`.
    pub fn json<T: Serialize + ?Sized>(mut self, value: &T) -> Result<Self, NetError> {
        self.body = serde_json::to_vec(value).map_err(|e| NetError::Encode(e.to_string()))?;
        self.headers.retain(|(k, _)| !k.eq_ignore_ascii_case("content-type"));
        self.headers.push(("content-type".into(), "application/json".into()));
        Ok(self)
    }
    /// Per-request timeout (covers connect, headers and body).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
    /// Per-request retry policy.
    pub fn retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = Some(retry);
        self
    }
}

/// A fully-buffered HTTP response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpResponse {
    /// Status code.
    pub status: u16,
    /// Response headers; names are lowercase.
    pub headers: Vec<(String, String)>,
    /// Raw body bytes.
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// `true` for 2xx statuses.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }
    /// First header value with this (case-insensitive) name.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    /// Body as UTF-8 (invalid sequences replaced).
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
    /// Deserialize the body as JSON.
    pub fn json<T: DeserializeOwned>(&self) -> Result<T, NetError> {
        serde_json::from_slice(&self.body).map_err(|e| NetError::Decode(e.to_string()))
    }
    /// Turn non-2xx responses into [`NetError::Status`].
    pub fn error_for_status(self) -> Result<Self, NetError> {
        if self.is_success() {
            Ok(self)
        } else {
            let mut body = self.text();
            if body.len() > 512 {
                let mut cut = 512;
                while !body.is_char_boundary(cut) {
                    cut -= 1;
                }
                body.truncate(cut);
            }
            Err(NetError::Status {
                status: self.status,
                body,
            })
        }
    }
}

/// Defaults applied to every request made through an [`Http`] instance.
#[derive(Clone, Debug, PartialEq)]
pub struct HttpConfig {
    /// Base that relative request URLs are joined onto (standard URL-join rules:
    /// use a trailing slash, e.g. `https://api.example.com/v1/`).
    pub base_url: Option<String>,
    /// Default timeout for the whole request (default 30 s).
    pub timeout: Option<Duration>,
    /// Headers sent with every request.
    pub headers: Vec<(String, String)>,
    /// Default retry policy (default: none).
    pub retry: RetryPolicy,
    /// `User-Agent` for native requests (browsers set their own).
    pub user_agent: String,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            base_url: None,
            timeout: Some(Duration::from_secs(30)),
            headers: Vec::new(),
            retry: RetryPolicy::none(),
            user_agent: concat!("space3d/", env!("CARGO_PKG_VERSION")).to_string(),
        }
    }
}

type Callback = Box<dyn FnMut(&mut SystemCtx) -> bool + Send>;

/// Non-blocking HTTP client resource, inserted by [`crate::NetPlugin`].
///
/// Cloning is cheap and clones share the connection pool and the callback queue.
///
/// ```no_run
/// use space3d_core::prelude::*;
/// use space3d_net::{Http, NetPlugin};
///
/// let mut app = App::new();
/// app.add_plugin(NetPlugin::default());
/// app.resources.expect::<Http>().get_then("https://example.com/scores.json", |ctx, result| {
///     match result {
///         Ok(resp) => println!("got {} bytes", resp.body.len()),
///         Err(e) => eprintln!("request failed: {e}"),
///     }
///     let _ = ctx; // full world/resources access on the main thread
/// });
/// ```
#[derive(Clone)]
pub struct Http {
    client: reqwest::Client,
    config: Arc<HttpConfig>,
    callbacks: Arc<Mutex<Vec<Callback>>>,
}

impl Default for Http {
    fn default() -> Self {
        Self::new()
    }
}

impl Http {
    /// Client with [`HttpConfig::default`].
    pub fn new() -> Self {
        Self::with_config(HttpConfig::default())
    }

    /// Client with custom defaults.
    pub fn with_config(config: HttpConfig) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let client = crate::tls::http_client_builder()
            .user_agent(config.user_agent.clone())
            .tcp_nodelay(true)
            // Drop idle pooled connections before servers close them (`HttpServer` closes idle
            // keep-alive connections after its 15 s header-read timeout), so a request never
            // races the server's close.
            .pool_idle_timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_else(|e| {
                tracing::error!("failed to build configured http client ({e}); using defaults");
                reqwest::Client::new()
            });
        #[cfg(target_arch = "wasm32")]
        let client = reqwest::Client::new();
        Self {
            client,
            config: Arc::new(config),
            callbacks: Arc::default(),
        }
    }

    /// The active defaults.
    pub fn config(&self) -> &HttpConfig {
        &self.config
    }

    /// Start `req` in the background. Never blocks; poll the returned task.
    pub fn request(&self, req: HttpRequest) -> Task<Result<HttpResponse, NetError>> {
        spawn(self.fetch(req))
    }

    /// The raw future behind [`Self::request`], for composing inside other async
    /// code that already runs on the engine runtime (e.g. via [`space3d_tasks::spawn`]).
    pub fn fetch(
        &self,
        req: HttpRequest,
    ) -> impl Future<Output = Result<HttpResponse, NetError>> + MaybeSend + 'static + use<> {
        let client = self.client.clone();
        let config = self.config.clone();
        async move { execute(&client, &config, &req).await }
    }

    /// `GET url`.
    pub fn get(&self, url: impl Into<String>) -> Task<Result<HttpResponse, NetError>> {
        self.request(HttpRequest::get(url))
    }
    /// `POST url` with a raw body.
    pub fn post(&self, url: impl Into<String>, body: impl Into<Vec<u8>>) -> Task<Result<HttpResponse, NetError>> {
        self.request(HttpRequest::post(url).body(body))
    }
    /// `PUT url` with a raw body.
    pub fn put(&self, url: impl Into<String>, body: impl Into<Vec<u8>>) -> Task<Result<HttpResponse, NetError>> {
        self.request(HttpRequest::put(url).body(body))
    }
    /// `DELETE url`.
    pub fn delete(&self, url: impl Into<String>) -> Task<Result<HttpResponse, NetError>> {
        self.request(HttpRequest::delete(url))
    }

    /// `GET url` and decode a 2xx JSON body into `T` (non-2xx → [`NetError::Status`]).
    pub fn get_json<T>(&self, url: impl Into<String>) -> Task<Result<T, NetError>>
    where
        T: DeserializeOwned + MaybeSend + 'static,
    {
        self.request_json(HttpRequest::get(url))
    }

    /// `POST` `body` as JSON and decode a 2xx JSON response into `R`.
    pub fn post_json<B, R>(&self, url: impl Into<String>, body: &B) -> Task<Result<R, NetError>>
    where
        B: Serialize + ?Sized,
        R: DeserializeOwned + MaybeSend + 'static,
    {
        match HttpRequest::post(url).json(body) {
            Ok(req) => self.request_json(req),
            Err(e) => spawn(async move { Err(e) }),
        }
    }

    /// Run any request and decode a 2xx JSON response into `T`.
    pub fn request_json<T>(&self, req: HttpRequest) -> Task<Result<T, NetError>>
    where
        T: DeserializeOwned + MaybeSend + 'static,
    {
        let fut = self.fetch(req);
        spawn(async move { fut.await?.error_for_status()?.json::<T>() })
    }

    /// Start `req` and run `then` on the main thread (in `PreUpdate`) when it finishes.
    ///
    /// Requires [`crate::NetPlugin`], which drains completed callbacks each frame.
    pub fn request_then(
        &self,
        req: HttpRequest,
        then: impl FnOnce(&mut SystemCtx, Result<HttpResponse, NetError>) + Send + 'static,
    ) {
        self.track(self.request(req), then);
    }

    /// `GET url`, then run `then(ctx, result)` on the main thread.
    pub fn get_then(
        &self,
        url: impl Into<String>,
        then: impl FnOnce(&mut SystemCtx, Result<HttpResponse, NetError>) + Send + 'static,
    ) {
        self.request_then(HttpRequest::get(url), then);
    }

    /// `POST url` with a raw body, then run `then(ctx, result)` on the main thread.
    pub fn post_then(
        &self,
        url: impl Into<String>,
        body: impl Into<Vec<u8>>,
        then: impl FnOnce(&mut SystemCtx, Result<HttpResponse, NetError>) + Send + 'static,
    ) {
        self.request_then(HttpRequest::post(url).body(body), then);
    }

    /// `GET url` as JSON, then run `then(ctx, result)` on the main thread.
    pub fn get_json_then<T>(
        &self,
        url: impl Into<String>,
        then: impl FnOnce(&mut SystemCtx, Result<T, NetError>) + Send + 'static,
    ) where
        T: DeserializeOwned + Send + 'static,
    {
        self.track(self.get_json(url), then);
    }

    /// Attach a main-thread callback to any task (runs in `PreUpdate` via [`crate::NetPlugin`]).
    pub fn track<T: Send + 'static>(&self, mut task: Task<T>, then: impl FnOnce(&mut SystemCtx, T) + Send + 'static) {
        let mut then = Some(then);
        self.callbacks.lock().push(Box::new(move |ctx| match task.state() {
            TaskState::Pending => false,
            TaskState::Cancelled => true,
            TaskState::Ready => {
                if let (Some(v), Some(f)) = (task.poll(), then.take()) {
                    f(ctx, v);
                }
                true
            }
        }));
    }

    /// Number of `*_then` callbacks still waiting for their request.
    pub fn pending_callbacks(&self) -> usize {
        self.callbacks.lock().len()
    }

    /// Run completed callbacks. Called by [`crate::NetPlugin`] in `PreUpdate`.
    pub(crate) fn run_callbacks(ctx: &mut SystemCtx) {
        let Some(queue) = ctx.resources.get::<Http>().map(|h| h.callbacks.clone()) else {
            return;
        };
        let mut pending = std::mem::take(&mut *queue.lock());
        if pending.is_empty() {
            return;
        }
        pending.retain_mut(|cb| !cb(ctx));
        // Callbacks may have queued new requests; keep both, oldest first.
        let mut q = queue.lock();
        pending.append(&mut q);
        *q = pending;
    }
}

fn resolve_url(config: &HttpConfig, raw: &str) -> Result<url::Url, NetError> {
    match url::Url::parse(raw) {
        Ok(u) => Ok(u),
        Err(url::ParseError::RelativeUrlWithoutBase) => {
            let base = config
                .base_url
                .clone()
                .or_else(platform_base_url)
                .ok_or_else(|| NetError::InvalidUrl(format!("{raw} (relative url and no base_url configured)")))?;
            url::Url::parse(&base)
                .and_then(|b| b.join(raw))
                .map_err(|e| NetError::InvalidUrl(format!("{raw}: {e}")))
        }
        Err(e) => Err(NetError::InvalidUrl(format!("{raw}: {e}"))),
    }
}

/// The page URL on the web, so `"/api/x"` works like in JavaScript.
#[cfg(target_arch = "wasm32")]
pub(crate) fn platform_base_url() -> Option<String> {
    let location = js_sys::Reflect::get(&js_sys::global(), &"location".into()).ok()?;
    js_sys::Reflect::get(&location, &"href".into()).ok()?.as_string()
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn platform_base_url() -> Option<String> {
    None
}

fn map_reqwest(e: reqwest::Error) -> NetError {
    let msg = error_chain(&e);
    if e.is_timeout() {
        return NetError::Timeout;
    }
    if e.is_builder() {
        return NetError::InvalidUrl(msg);
    }
    #[cfg(not(target_arch = "wasm32"))]
    if e.is_connect() {
        return NetError::Connect(msg);
    }
    if e.is_decode() || e.is_body() {
        return NetError::Transport(msg);
    }
    #[cfg(target_arch = "wasm32")]
    if e.is_request() {
        // fetch() rejects with a TypeError for network/CORS failures.
        return NetError::Connect(msg);
    }
    NetError::Transport(msg)
}

async fn execute_once(
    client: &reqwest::Client,
    config: &HttpConfig,
    url: &url::Url,
    req: &HttpRequest,
) -> Result<HttpResponse, NetError> {
    let mut rb = client.request(req.method.clone(), url.clone());
    for (k, v) in &config.headers {
        if !req.headers.iter().any(|(rk, _)| rk.eq_ignore_ascii_case(k)) {
            rb = rb.header(k.as_str(), v.as_str());
        }
    }
    for (k, v) in &req.headers {
        rb = rb.header(k.as_str(), v.as_str());
    }
    if !req.body.is_empty() {
        rb = rb.body(req.body.clone());
    }
    let resp = rb.send().await.map_err(map_reqwest)?;
    let status = resp.status().as_u16();
    let headers = resp
        .headers()
        .iter()
        .map(|(k, v)| {
            (
                k.as_str().to_owned(),
                String::from_utf8_lossy(v.as_bytes()).into_owned(),
            )
        })
        .collect();
    let body = resp.bytes().await.map_err(map_reqwest)?.to_vec();
    Ok(HttpResponse { status, headers, body })
}

async fn execute(client: &reqwest::Client, config: &HttpConfig, req: &HttpRequest) -> Result<HttpResponse, NetError> {
    let url = resolve_url(config, &req.url)?;
    let retry = req.retry.unwrap_or(config.retry);
    let limit = req.timeout.or(config.timeout);
    let mut attempt = 0u32;
    loop {
        let result = rt::timeout(limit, execute_once(client, config, &url, req))
            .await
            .and_then(|r| r);
        let retry_this = match &result {
            Ok(resp) => retry.should_retry_status(resp.status),
            Err(e) => e.is_retryable(),
        };
        if !retry_this || attempt >= retry.max_retries {
            return result;
        }
        let delay = retry.backoff.delay(attempt);
        tracing::debug!(url = %url, attempt, ?delay, "retrying http request");
        rt::sleep(delay).await;
        attempt += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_builder() {
        let r = HttpRequest::post("http://x/y")
            .header("X-A", "1")
            .json(&serde_json::json!({"a": 1}))
            .unwrap()
            .timeout(Duration::from_secs(1));
        assert_eq!(r.method, Method::POST);
        assert_eq!(r.body, br#"{"a":1}"#);
        assert!(
            r.headers
                .iter()
                .any(|(k, v)| k == "content-type" && v == "application/json")
        );
        assert_eq!(r.timeout, Some(Duration::from_secs(1)));
    }

    #[test]
    fn response_helpers() {
        let r = HttpResponse {
            status: 404,
            headers: vec![("content-type".into(), "text/plain".into())],
            body: b"nope".to_vec(),
        };
        assert_eq!(r.header("Content-Type"), Some("text/plain"));
        assert!(!r.is_success());
        assert_eq!(
            r.clone().error_for_status(),
            Err(NetError::Status {
                status: 404,
                body: "nope".into()
            })
        );
        let ok = HttpResponse {
            status: 200,
            headers: vec![],
            body: br#"[1,2]"#.to_vec(),
        };
        assert_eq!(ok.json::<Vec<u8>>().unwrap(), vec![1, 2]);
    }

    #[test]
    fn url_resolution() {
        let mut cfg = HttpConfig::default();
        assert!(resolve_url(&cfg, "/relative").is_err());
        cfg.base_url = Some("http://localhost:8080/api/".into());
        assert_eq!(
            resolve_url(&cfg, "users").unwrap().as_str(),
            "http://localhost:8080/api/users"
        );
        assert_eq!(
            resolve_url(&cfg, "https://other/x").unwrap().as_str(),
            "https://other/x"
        );
    }
}
