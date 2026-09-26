//! The loopback server that serves the release UI at `http://localhost:<port>`
//! (why it exists: AGENTS.md §2, cross-origin isolation).
//!
//! That origin is granted every app command (capabilities/default.json), so
//! whoever answers on the port controls Andai. `tauri-plugin-localhost` bound
//! the port in a background thread and only panicked there if it was taken:
//! the window still opened and loaded *another process's* page with full IPC.
//! Here the bind happens before any window exists, on both loopback families,
//! and a failed bind means no window (AGENTS.md §9).

use std::io::ErrorKind;
use std::net::{Ipv4Addr, Ipv6Addr, TcpListener};
use tauri::{Runtime, Url};

/// One embedded asset, as the resolver hands it over.
pub struct Page {
    pub bytes: Vec<u8>,
    pub mime: String,
    pub csp: Option<String>,
}

#[derive(Debug)]
pub struct Reply {
    pub status: u16,
    pub headers: Vec<(&'static str, String)>,
    pub body: Vec<u8>,
}

/// Binds `127.0.0.1:<port>` and `[::1]:<port>`. WebKit may resolve
/// `localhost` to either, so both must be ours or nobody can be trusted with
/// it. A Mac without IPv6 loopback has nothing to squat there, so only that
/// case is skipped.
pub fn bind(port: u16) -> Result<Vec<TcpListener>, String> {
    let in_use = |e: std::io::Error| {
        format!(
            "Port {port} is already in use by another app ({e}). Andai only loads its interface from a port it \
             owns. Quit the app using it (see `lsof -i :{port}` in Terminal), then open Andai again."
        )
    };
    let mut listeners = vec![TcpListener::bind((Ipv4Addr::LOCALHOST, port)).map_err(in_use)?];
    match TcpListener::bind((Ipv6Addr::LOCALHOST, port)) {
        Ok(l) => listeners.push(l),
        Err(e) if e.kind() == ErrorKind::AddrNotAvailable => {}
        Err(e) => return Err(in_use(e)),
    }
    Ok(listeners)
}

/// Decides the response for one request. Pure, so the policy is unit-tested.
///
/// - Only `Host: localhost:<port>` is answered. A web page in the user's
///   browser can point its own domain at 127.0.0.1 (DNS rebinding); its
///   requests carry that domain as Host and get 403.
/// - Only embedded assets are served, never the filesystem.
pub fn respond(port: u16, host: Option<&str>, url: &str, lookup: impl Fn(&str) -> Option<Page>) -> Reply {
    let reply = |status, body: &str| Reply { status, headers: vec![], body: body.as_bytes().to_vec() };
    if !host.is_some_and(|h| h.eq_ignore_ascii_case(&format!("localhost:{port}"))) {
        return reply(403, "Forbidden");
    }
    let path = url.split(['?', '#']).next().unwrap_or("/");
    let Some(page) = lookup(path) else { return reply(404, "Not found") };
    let mut headers = vec![
        ("Content-Type", page.mime),
        ("Cache-Control", "no-cache".into()),
        // Cross-origin isolation for multi-threaded wllama (AGENTS.md §2).
        ("Cross-Origin-Opener-Policy", "same-origin".into()),
        ("Cross-Origin-Embedder-Policy", "require-corp".into()),
        ("Cross-Origin-Resource-Policy", "cross-origin".into()),
        ("X-Content-Type-Options", "nosniff".into()),
        ("Referrer-Policy", "no-referrer".into()),
    ];
    if let Some(csp) = page.csp {
        headers.push(("Content-Security-Policy", csp));
    }
    Reply { status: 200, headers, body: page.bytes }
}

/// Serves the embedded assets on every listener, one thread each.
pub fn serve<R: Runtime>(listeners: Vec<TcpListener>, port: u16, assets: tauri::AssetResolver<R>) {
    // AssetResolver's derived Clone needs `R: Clone`; share it instead.
    let assets = std::sync::Arc::new(assets);
    for listener in listeners {
        let assets = assets.clone();
        let lookup = move |path: &str| {
            assets.get(path.to_string()).map(|a| Page { bytes: a.bytes, mime: a.mime_type, csp: a.csp_header })
        };
        run(listener, port, lookup);
    }
}

fn run(listener: TcpListener, port: u16, lookup: impl Fn(&str) -> Option<Page> + Send + 'static) {
    let server = tiny_http::Server::from_listener(listener, None).expect("listener is bound");
    std::thread::spawn(move || {
        for req in server.incoming_requests() {
            let host = req.headers().iter().find(|h| h.field.equiv("Host")).map(|h| h.value.as_str().to_string());
            let reply = respond(port, host.as_deref(), req.url(), &lookup);
            let mut resp = tiny_http::Response::from_data(reply.body).with_status_code(reply.status);
            for (name, value) in reply.headers {
                if let Ok(h) = tiny_http::Header::from_bytes(name.as_bytes(), value.as_bytes()) {
                    resp.add_header(h);
                }
            }
            let _ = req.respond(resp);
        }
    });
}

/// The webview may only ever show the app itself: top-level navigation to any
/// other origin (a link in model output, an injected `location.href`) is
/// refused, so no foreign page can render inside Andai's window.
pub fn is_app_url(url: &Url, app: &Url) -> bool {
    url.scheme() == app.scheme() && url.host_str() == app.host_str() && url.port_or_known_default() == app.port_or_known_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;

    fn page(path: &str) -> Option<Page> {
        (path == "/index.html" || path == "/").then(|| Page {
            bytes: b"<html>".to_vec(),
            mime: "text/html".into(),
            csp: Some("default-src 'self'".into()),
        })
    }

    fn header<'a>(r: &'a Reply, name: &str) -> Option<&'a str> {
        r.headers.iter().find(|(n, _)| *n == name).map(|(_, v)| v.as_str())
    }

    #[test]
    fn a_taken_port_refuses_to_start() {
        let squatter = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = squatter.local_addr().unwrap().port();
        let err = bind(port).unwrap_err();
        assert!(err.contains("already in use"), "{err}");
    }

    #[test]
    fn a_squatter_on_ipv6_loopback_also_refuses() {
        let Ok(squatter) = TcpListener::bind((Ipv6Addr::LOCALHOST, 0)) else { return };
        let port = squatter.local_addr().unwrap().port();
        assert!(bind(port).is_err());
    }

    #[test]
    fn a_free_port_binds_both_families() {
        let port = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap().local_addr().unwrap().port();
        let listeners = bind(port).unwrap();
        assert!(!listeners.is_empty());
        assert!(TcpListener::bind((Ipv4Addr::LOCALHOST, port)).is_err(), "port is held while Andai runs");
    }

    #[test]
    fn foreign_hosts_are_forbidden() {
        for host in [None, Some("evil.example:14230"), Some("127.0.0.1:14230"), Some("localhost:1")] {
            assert_eq!(respond(14230, host, "/", page).status, 403, "{host:?}");
        }
    }

    #[test]
    fn assets_carry_isolation_and_hardening_headers() {
        let r = respond(14230, Some("localhost:14230"), "/index.html?x=1#y", page);
        assert_eq!(r.status, 200);
        assert_eq!(header(&r, "Cross-Origin-Embedder-Policy"), Some("require-corp"));
        assert_eq!(header(&r, "Cross-Origin-Opener-Policy"), Some("same-origin"));
        assert_eq!(header(&r, "X-Content-Type-Options"), Some("nosniff"));
        assert_eq!(header(&r, "Referrer-Policy"), Some("no-referrer"));
        assert_eq!(header(&r, "Content-Security-Policy"), Some("default-src 'self'"));
    }

    #[test]
    fn unknown_paths_are_not_found() {
        assert_eq!(respond(14230, Some("localhost:14230"), "/../../etc/passwd", page).status, 404);
    }

    #[test]
    fn serves_over_real_sockets() {
        let port = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap().local_addr().unwrap().port();
        for l in bind(port).unwrap() {
            run(l, port, page);
        }
        let get = |host: &str| {
            let mut s = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).unwrap();
            write!(s, "GET / HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n").unwrap();
            let mut out = String::new();
            s.read_to_string(&mut out).unwrap();
            out
        };
        assert!(get(&format!("localhost:{port}")).starts_with("HTTP/1.1 200"));
        assert!(get("rebind.evil.example").starts_with("HTTP/1.1 403"));
    }

    #[test]
    fn navigation_is_limited_to_the_app_origin() {
        let app: Url = "http://localhost:14230".parse().unwrap();
        let ok = |u: &str| is_app_url(&u.parse().unwrap(), &app);
        assert!(ok("http://localhost:14230/#knowledge"));
        assert!(!ok("https://example.com/"));
        assert!(!ok("http://localhost:1420/"));
        assert!(!ok("http://127.0.0.1:14230/"));
        assert!(!ok("file:///etc/passwd"));
    }
}
