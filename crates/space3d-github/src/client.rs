use crate::error::{GithubError, Result};
use crate::models::{
    CreateRepo, Issue, NewIssue, NewRelease, Release, ReleaseAsset, RepoRef, Repository, WorkflowRun, WorkflowRunsPage,
    WorkflowRunsQuery,
};
use reqwest::header::{ACCEPT, CONTENT_TYPE};
use reqwest::{Method, RequestBuilder, Response};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use space3d_tasks::{MaybeSend, Task};
use std::fmt;
use std::future::Future;
use std::sync::Arc;

/// Public GitHub REST API root.
pub const DEFAULT_API_URL: &str = "https://api.github.com";
/// REST API version sent as `X-GitHub-Api-Version`.
pub const API_VERSION: &str = "2022-11-28";
/// Media type sent as `Accept`.
pub const ACCEPT_GITHUB_JSON: &str = "application/vnd.github+json";

/// Thin async client for the GitHub REST API.
///
/// * From the **CLI / tools**: call the `async fn`s directly.
/// * From **engine systems**: wrap calls with [`GithubClient::task`] (or the
///   `*_task` helpers) to get a pollable [`Task`] that never blocks the frame.
///
/// The API root is injectable ([`GithubClient::with_base_url`]) so tests can
/// point it at a local mock server, and games can route requests through their
/// own proxy (recommended for crash reports: never ship a personal token).
#[derive(Clone)]
pub struct GithubClient {
    http: reqwest::Client,
    base_url: Arc<str>,
    token: Option<Arc<str>>,
}

impl fmt::Debug for GithubClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GithubClient")
            .field("base_url", &self.base_url)
            .field("token", &self.token.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

impl Default for GithubClient {
    fn default() -> Self {
        Self::new(None)
    }
}

impl GithubClient {
    /// Client for `api.github.com` with an optional token.
    pub fn new(token: Option<String>) -> Self {
        #[cfg(not(target_os = "android"))]
        let builder = reqwest::Client::builder();
        // Android: HTTPS checked against the compiled-in roots (see `space3d_net::http_client_builder`).
        #[cfg(target_os = "android")]
        let builder = space3d_net::http_client_builder();
        #[cfg(not(target_arch = "wasm32"))]
        let builder = builder.user_agent(concat!("space3d/", env!("CARGO_PKG_VERSION")));
        Self {
            http: builder.build().expect("failed to build HTTP client"),
            base_url: DEFAULT_API_URL.into(),
            token: token.filter(|t| !t.is_empty()).map(Into::into),
        }
    }

    /// Client using [`discover_token`] (`GITHUB_TOKEN`, `GH_TOKEN`, then `gh auth token`).
    ///
    /// Runs the `gh` CLI synchronously if no env var is set — call it from
    /// tools or at startup, not from a system.
    pub fn from_env() -> Self {
        Self::new(discover_token())
    }

    /// Point at a different API root (GitHub Enterprise, a proxy, or a test server).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').into();
        self
    }

    /// Replace the token.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into().into());
        self
    }

    /// Use a preconfigured `reqwest` client (timeouts, proxies, ...).
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// API root in use.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// `true` if requests are authenticated.
    pub fn has_token(&self) -> bool {
        self.token.is_some()
    }

    /// Run an async operation on the background runtime and get a pollable [`Task`].
    ///
    /// ```no_run
    /// # use space3d_github::{GithubClient, RepoRef};
    /// let client = GithubClient::from_env();
    /// let repo = RepoRef::new("SpaceCorps", "Space3d");
    /// let mut task = client.task(move |c| async move { c.list_releases(&repo).await });
    /// // later, in a system:
    /// if let Some(Ok(releases)) = task.poll() { println!("{} releases", releases.len()); }
    /// ```
    pub fn task<T, F, Fut>(&self, f: F) -> Task<Result<T>>
    where
        T: MaybeSend + 'static,
        F: FnOnce(GithubClient) -> Fut,
        Fut: Future<Output = Result<T>> + MaybeSend + 'static,
    {
        space3d_tasks::spawn(f(self.clone()))
    }

    // ---- plumbing -------------------------------------------------------

    fn request(&self, method: Method, url: &str) -> RequestBuilder {
        let url = if url.starts_with("http://") || url.starts_with("https://") {
            url.to_owned()
        } else {
            format!("{}{url}", self.base_url)
        };
        let rb = self
            .http
            .request(method, url)
            .header(ACCEPT, ACCEPT_GITHUB_JSON)
            .header("X-GitHub-Api-Version", API_VERSION);
        match &self.token {
            Some(t) => rb.bearer_auth(t),
            None => rb,
        }
    }

    /// Writes against real GitHub always need a token; custom API roots (proxies,
    /// mock servers) decide for themselves.
    fn ensure_can_write(&self) -> Result<()> {
        if self.token.is_none() && &*self.base_url == DEFAULT_API_URL {
            Err(GithubError::NoToken)
        } else {
            Ok(())
        }
    }

    async fn checked(rb: RequestBuilder) -> Result<Response> {
        let resp = rb.send().await?;
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        #[derive(Deserialize)]
        struct ErrorBody {
            message: String,
            documentation_url: Option<String>,
            #[serde(default)]
            errors: Option<serde_json::Value>,
        }
        let text = resp.text().await.unwrap_or_default();
        let (message, documentation_url) = match serde_json::from_str::<ErrorBody>(&text) {
            Ok(b) => (
                match b.errors {
                    Some(errs) => format!("{} ({errs})", b.message),
                    None => b.message,
                },
                b.documentation_url,
            ),
            Err(_) if text.is_empty() => (status.canonical_reason().unwrap_or("error").to_owned(), None),
            Err(_) => (text, None),
        };
        Err(GithubError::Api {
            status: status.as_u16(),
            message,
            documentation_url,
        })
    }

    async fn send_json<T: DeserializeOwned>(rb: RequestBuilder) -> Result<T> {
        let bytes = Self::checked(rb).await?.bytes().await?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    fn not_found_to_none<T>(r: Result<T>) -> Result<Option<T>> {
        match r {
            Ok(v) => Ok(Some(v)),
            Err(e) if e.is_not_found() => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn repo_path(repo: &RepoRef) -> String {
        format!("/repos/{}/{}", encode_segment(&repo.owner), encode_segment(&repo.name))
    }

    // ---- repos ----------------------------------------------------------

    /// `GET /repos/{owner}/{repo}`.
    pub async fn get_repo(&self, repo: &RepoRef) -> Result<Repository> {
        Self::send_json(self.request(Method::GET, &Self::repo_path(repo))).await
    }

    /// `POST /user/repos` or `POST /orgs/{org}/repos`.
    pub async fn create_repo(&self, req: &CreateRepo) -> Result<Repository> {
        self.ensure_can_write()?;
        let path = match &req.org {
            Some(org) => format!("/orgs/{}/repos", encode_segment(org)),
            None => "/user/repos".to_owned(),
        };
        Self::send_json(self.request(Method::POST, &path).json(req)).await
    }

    // ---- releases -------------------------------------------------------

    /// `GET /repos/{owner}/{repo}/releases` (first 100, newest first).
    pub async fn list_releases(&self, repo: &RepoRef) -> Result<Vec<Release>> {
        let rb = self
            .request(Method::GET, &format!("{}/releases", Self::repo_path(repo)))
            .query(&[("per_page", "100")]);
        Self::send_json(rb).await
    }

    /// `GET /repos/{owner}/{repo}/releases/latest` — newest non-draft,
    /// non-prerelease release; `None` if there is none.
    pub async fn latest_release(&self, repo: &RepoRef) -> Result<Option<Release>> {
        let rb = self.request(Method::GET, &format!("{}/releases/latest", Self::repo_path(repo)));
        Self::not_found_to_none(Self::send_json(rb).await)
    }

    /// `GET /repos/{owner}/{repo}/releases/tags/{tag}`; `None` if missing.
    pub async fn release_by_tag(&self, repo: &RepoRef, tag: &str) -> Result<Option<Release>> {
        let rb = self.request(
            Method::GET,
            &format!("{}/releases/tags/{}", Self::repo_path(repo), encode_segment(tag)),
        );
        Self::not_found_to_none(Self::send_json(rb).await)
    }

    /// `POST /repos/{owner}/{repo}/releases`.
    pub async fn create_release(&self, repo: &RepoRef, release: &NewRelease) -> Result<Release> {
        self.ensure_can_write()?;
        let rb = self
            .request(Method::POST, &format!("{}/releases", Self::repo_path(repo)))
            .json(release);
        Self::send_json(rb).await
    }

    /// Upload a build (zip, wasm bundle, apk, ipa, ...) to `release` using its
    /// `upload_url`. See [`crate::content_type_for`] for a MIME type.
    pub async fn upload_release_asset(
        &self,
        release: &Release,
        name: &str,
        content_type: &str,
        data: Vec<u8>,
    ) -> Result<ReleaseAsset> {
        self.ensure_can_write()?;
        let url = release.upload_url.split('{').next().unwrap_or_default();
        if url.is_empty() {
            return Err(GithubError::Api {
                status: 0,
                message: format!("release {} has no upload_url", release.tag_name),
                documentation_url: None,
            });
        }
        let rb = self
            .request(Method::POST, url)
            .query(&[("name", name)])
            .header(CONTENT_TYPE, content_type)
            .body(data);
        Self::send_json(rb).await
    }

    /// Read `path` and upload it under its file name (native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn upload_release_asset_file(
        &self,
        release: &Release,
        path: impl AsRef<std::path::Path>,
    ) -> Result<ReleaseAsset> {
        let path = path.as_ref();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "asset path has no file name"))?
            .to_owned();
        let data = tokio::fs::read(path).await?;
        self.upload_release_asset(release, &name, crate::content_type_for(&name), data)
            .await
    }

    /// `DELETE /repos/{owner}/{repo}/releases/assets/{id}` (e.g. to replace an asset).
    pub async fn delete_release_asset(&self, repo: &RepoRef, asset_id: u64) -> Result<()> {
        self.ensure_can_write()?;
        let rb = self.request(
            Method::DELETE,
            &format!("{}/releases/assets/{asset_id}", Self::repo_path(repo)),
        );
        Self::checked(rb).await.map(drop)
    }

    // ---- issues ---------------------------------------------------------

    /// `POST /repos/{owner}/{repo}/issues`.
    pub async fn create_issue(&self, repo: &RepoRef, issue: &NewIssue) -> Result<Issue> {
        self.ensure_can_write()?;
        let rb = self
            .request(Method::POST, &format!("{}/issues", Self::repo_path(repo)))
            .json(issue);
        Self::send_json(rb).await
    }

    // ---- actions --------------------------------------------------------

    /// `POST /repos/{owner}/{repo}/actions/workflows/{workflow}/dispatches` —
    /// trigger a `workflow_dispatch` run (e.g. a CI build) on `git_ref`.
    /// `workflow` is a file name (`space3d.yml`) or numeric id.
    pub async fn dispatch_workflow(
        &self,
        repo: &RepoRef,
        workflow: &str,
        git_ref: &str,
        inputs: Option<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<()> {
        self.ensure_can_write()?;
        let mut body = serde_json::json!({ "ref": git_ref });
        if let Some(inputs) = inputs {
            body["inputs"] = serde_json::Value::Object(inputs);
        }
        let rb = self
            .request(
                Method::POST,
                &format!(
                    "{}/actions/workflows/{}/dispatches",
                    Self::repo_path(repo),
                    encode_segment(workflow)
                ),
            )
            .json(&body);
        Self::checked(rb).await.map(drop)
    }

    /// `GET /repos/{owner}/{repo}/actions/runs` (or `.../workflows/{id}/runs`), newest first.
    pub async fn list_workflow_runs(&self, repo: &RepoRef, query: &WorkflowRunsQuery) -> Result<Vec<WorkflowRun>> {
        let path = match &query.workflow {
            Some(w) => format!("{}/actions/workflows/{}/runs", Self::repo_path(repo), encode_segment(w)),
            None => format!("{}/actions/runs", Self::repo_path(repo)),
        };
        let mut params: Vec<(&str, String)> = Vec::new();
        if let Some(b) = &query.branch {
            params.push(("branch", b.clone()));
        }
        if let Some(s) = &query.status {
            params.push(("status", s.clone()));
        }
        if let Some(e) = &query.event {
            params.push(("event", e.clone()));
        }
        if let Some(n) = query.per_page {
            params.push(("per_page", n.to_string()));
        }
        let page: WorkflowRunsPage = Self::send_json(self.request(Method::GET, &path).query(&params)).await?;
        Ok(page.workflow_runs)
    }

    // ---- Task helpers for engine code ------------------------------------

    /// [`GithubClient::latest_release`] as a pollable [`Task`].
    pub fn latest_release_task(&self, repo: RepoRef) -> Task<Result<Option<Release>>> {
        self.task(move |c| async move { c.latest_release(&repo).await })
    }

    /// [`GithubClient::create_issue`] as a pollable [`Task`].
    pub fn create_issue_task(&self, repo: RepoRef, issue: NewIssue) -> Task<Result<Issue>> {
        self.task(move |c| async move { c.create_issue(&repo, &issue).await })
    }

    /// [`GithubClient::dispatch_workflow`] as a pollable [`Task`].
    pub fn dispatch_workflow_task(&self, repo: RepoRef, workflow: String, git_ref: String) -> Task<Result<()>> {
        self.task(move |c| async move { c.dispatch_workflow(&repo, &workflow, &git_ref, None).await })
    }
}

/// Percent-encode one URL path segment (everything but RFC 3986 unreserved chars).
fn encode_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Find a GitHub token: `GITHUB_TOKEN`, then `GH_TOKEN`, then (native only)
/// the output of `gh auth token`. Returns `None` if nothing is configured.
///
/// Spawns the `gh` process synchronously when no env var is set; use from
/// tools/CLI or startup code, never from a per-frame system.
pub fn discover_token() -> Option<String> {
    resolve_token(|k| std::env::var(k).ok(), gh_auth_token)
}

/// Like [`discover_token`] but never spawns `gh` (env vars only).
pub fn token_from_env() -> Option<String> {
    resolve_token(|k| std::env::var(k).ok(), || None)
}

fn resolve_token(env: impl Fn(&str) -> Option<String>, gh: impl FnOnce() -> Option<String>) -> Option<String> {
    let clean = |t: String| Some(t.trim().to_owned()).filter(|t| !t.is_empty());
    ["GITHUB_TOKEN", "GH_TOKEN"]
        .into_iter()
        .find_map(|k| env(k).and_then(clean))
        .or_else(|| gh().and_then(clean))
}

#[cfg(not(target_arch = "wasm32"))]
fn gh_auth_token() -> Option<String> {
    let out = std::process::Command::new("gh")
        .args(["auth", "token"])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8(out.stdout).ok())
        .flatten()
}

#[cfg(target_arch = "wasm32")]
fn gh_auth_token() -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_precedence() {
        let env = |k: &str| match k {
            "GH_TOKEN" => Some(" gh-env \n".to_string()),
            _ => None,
        };
        assert_eq!(resolve_token(env, || Some("cli".into())).as_deref(), Some("gh-env"));
        let env = |k: &str| (k == "GITHUB_TOKEN").then(|| "primary".to_string());
        assert_eq!(
            resolve_token(env, || panic!("gh must not run")).as_deref(),
            Some("primary")
        );
        let blank = |_: &str| Some("  ".to_string());
        assert_eq!(resolve_token(blank, || Some("cli\n".into())).as_deref(), Some("cli"));
        assert_eq!(resolve_token(|_| None, || None), None);
    }

    #[test]
    fn segments_are_encoded() {
        assert_eq!(encode_segment("v1.2.3"), "v1.2.3");
        assert_eq!(encode_segment("release/1 2"), "release%2F1%202");
    }

    #[test]
    fn writes_to_github_without_token_fail_fast() {
        let c = GithubClient::new(None);
        assert!(matches!(c.ensure_can_write(), Err(GithubError::NoToken)));
        assert!(
            c.clone()
                .with_base_url("http://127.0.0.1:1/")
                .ensure_can_write()
                .is_ok()
        );
        assert_eq!(c.with_base_url("http://x/").base_url(), "http://x");
        assert!(!format!("{:?}", GithubClient::new(Some("secret".into()))).contains("secret"));
    }
}
