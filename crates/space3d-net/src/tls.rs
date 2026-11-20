//! Process-wide rustls crypto provider (native only).

/// tokio-tungstenite builds its rustls config from the process default provider,
/// which rustls can only infer when exactly one backend is compiled in. Pin
/// aws-lc-rs (the one reqwest uses) so `wss://` keeps working if another crate
/// in the final binary also enables `ring`.
pub(crate) fn ensure_crypto_provider() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if rustls::crypto::CryptoProvider::get_default().is_none() {
            // Losing a race against another installer is fine: any provider works.
            let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        }
    });
}

/// A `reqwest` client builder whose HTTPS works on this platform; every native HTTP client of
/// the engine starts from it.
///
/// Android: reqwest's default certificate check (rustls-platform-verifier) needs a Java class
/// in the APK and a JNI call at start, which a NativeActivity app has not: its first HTTPS
/// request would panic. There the certificates are checked against the Mozilla roots compiled
/// in (webpki-roots), as the WebSocket client does on every platform.
pub fn http_client_builder() -> reqwest::ClientBuilder {
    let builder = reqwest::Client::builder();
    #[cfg(target_os = "android")]
    let builder = {
        ensure_crypto_provider();
        let mut roots = rustls::RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        let config = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        builder.use_preconfigured_tls(config)
    };
    builder
}
