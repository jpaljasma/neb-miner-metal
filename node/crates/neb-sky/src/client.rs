//! The miner side of the transport (`spec/sky-search.md` section 10): ask the batch server for a batch, run the
//! search, post the result. Plain HTTP; nothing secret travels and the server checks
//! every result. This port of the v1.0.11 client uses ureq for HTTP framing and
//! counts service failures separately from an empty queue or a refused result.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Longest reply read from the server.
const MOST_BYTES: u64 = 64 << 20;
/// Seconds between two tries when the server has no batch or does not answer.
const WAIT: u64 = 60;

/// Where the batches come from and what the miner says of itself. The three text fields are optional.
#[derive(Clone, Debug, Default)]
pub struct Config {
    /// The batch server, `http://host:port`.
    pub url: String,
    /// Name and version of the mining software.
    pub software: String,
    /// Model of the card or processor.
    pub device: String,
    /// A NEB address.
    pub address: String,
    /// Threads that run the search.
    pub threads: usize,
}

/// Counters of a running worker.
#[derive(Debug, Default)]
pub struct Stats {
    /// Results the server took.
    pub batches: AtomicU64,
    /// Light curves in those results.
    pub curves: AtomicU64,
    /// Results the server refused.
    pub refused: AtomicU64,
    /// Requests that failed.
    pub errors: AtomicU64,
}

/// Why a request failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The address is not `http://host[:port]`.
    Url(String),
    /// The connection failed or the reply was cut.
    Io(std::io::Error),
    /// The server returned an unexpected HTTP status.
    Status(u16),
    /// The server returned an invalid batch.
    Reply,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Url(u) => write!(f, "the sky server address must be http://host:port, got {u}"),
            Error::Io(e) => write!(f, "sky server: {e}"),
            Error::Status(code) => write!(f, "sky server returned HTTP {code}"),
            Error::Reply => f.write_str("the sky server sent an invalid batch"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// `host:port` and the path prefix of an `http://` address.
fn split(url: &str) -> Result<(String, String), Error> {
    let rest = url.strip_prefix("http://").ok_or_else(|| Error::Url(url.to_string()))?;
    let (host, prefix) = rest.split_once('/').map_or((rest, ""), |(h, p)| (h, p.trim_end_matches('/')));
    if host.is_empty() {
        return Err(Error::Url(url.to_string()));
    }
    let host = if host.contains(':') { host.to_string() } else { format!("{host}:80") };
    Ok((host, if prefix.is_empty() { String::new() } else { format!("/{prefix}") }))
}

/// A query value with every byte outside letters, digits, `-`, `.`, `_` and `~` percent-encoded.
fn encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// One request; the status code and the body of the reply.
fn request(agent: &ureq::Agent, config: &Config, route: &str, body: Option<&[u8]>) -> Result<(u16, Vec<u8>), Error> {
    let (host, prefix) = split(&config.url)?;
    let url = format!(
        "http://{host}{prefix}{route}?software={}&device={}&address={}",
        encode(&config.software),
        encode(&config.device),
        encode(&config.address)
    );
    let mut response = match body {
        None => agent.get(&url).call(),
        Some(data) => agent.post(&url).send(data),
    }
    .map_err(ureq::Error::into_io)?;
    let status = response.status().as_u16();
    let payload = response.body_mut().with_config().limit(MOST_BYTES).read_to_vec().map_err(ureq::Error::into_io)?;
    Ok((status, payload))
}

/// One round: a batch asked, searched and posted. `Ok(true)` when the server had a batch.
///
/// # Errors
/// When a request fails. A result the server refuses is counted, not an error.
pub fn round(config: &Config, stats: &Stats) -> Result<bool, Error> {
    let agent = ureq::Agent::config_builder()
        .proxy(None)
        .max_redirects(0)
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(20)))
        .timeout_global(Some(Duration::from_secs(120)))
        .build()
        .new_agent();
    let (status, raw) = request(&agent, config, "/sky/v1/batch", None)?;
    match status {
        200 => {}
        204 => return Ok(false),
        other => return Err(Error::Status(other)),
    }
    let Ok(result) = crate::run_threads(&raw, config.threads) else {
        return Err(Error::Reply);
    };
    let (status, _) = request(&agent, config, "/sky/v1/result", Some(&result))?;
    match status {
        200 => {
            stats.batches.fetch_add(1, Ordering::Relaxed);
            stats
                .curves
                .fetch_add(((result.len() - 44) / (crate::KEEP * crate::ENTRY_BYTES)) as u64, Ordering::Relaxed);
        }
        409 => {} // Another worker settled the batch.
        404 | 413 | 422 => {
            stats.refused.fetch_add(1, Ordering::Relaxed);
        }
        other => return Err(Error::Status(other)),
    }
    Ok(true)
}

/// Starts a thread that runs rounds until `stop` is set. It waits a minute when the server has no batch or does
/// not answer: a server that is away never slows mining down.
pub fn spawn(config: Config, stop: Arc<AtomicBool>) -> (JoinHandle<()>, Arc<Stats>) {
    let stats = Arc::new(Stats::default());
    let shared = stats.clone();
    let handle = thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            let busy = round(&config, &shared).unwrap_or_else(|_| {
                shared.errors.fetch_add(1, Ordering::Relaxed);
                false
            });
            for _ in 0..if busy { 0 } else { WAIT } {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                thread::sleep(Duration::from_secs(1));
            }
        }
    });
    (handle, stats)
}

/// The options of the sky search on a command line: `--sky URL`, `--sky-threads N`, `--sky-address ADDRESS` and
/// `--sky-device TEXT`.
#[derive(Clone, Debug, Default)]
pub struct Options {
    url: Option<String>,
    threads: usize,
    address: String,
    device: Option<String>,
}

impl Options {
    /// Takes one option and its value: `true` when it is a valid option of the sky search.
    pub fn take(&mut self, flag: &str, value: &str) -> bool {
        match flag {
            "--sky" => self.url = Some(value.to_string()),
            "--sky-address" => self.address = value.to_string(),
            "--sky-device" => self.device = Some(value.to_string()),
            "--sky-threads" => match value.parse::<usize>() {
                Ok(n) if n > 0 => self.threads = n,
                _ => return false,
            },
            _ => return false,
        }
        true
    }

    /// The settings of the worker, or `None` when `--sky` was not given. `device` is what the server is told when
    /// `--sky-device` is absent, `threads` how many threads search when `--sky-threads` is absent.
    ///
    /// # Errors
    /// When the address is not `http://host[:port]`.
    pub fn config(self, software: &str, device: &str, threads: usize) -> Result<Option<Config>, Error> {
        let Some(url) = self.url else { return Ok(None) };
        split(&url)?;
        Ok(Some(Config {
            url,
            software: software.to_string(),
            device: self.device.unwrap_or_else(|| device.to_string()),
            address: self.address,
            threads: if self.threads > 0 { self.threads } else { threads.max(1) },
        }))
    }

    /// Starts the worker when `--sky` was given; one thread unless `--sky-threads` says otherwise.
    ///
    /// # Errors
    /// When the address is not `http://host[:port]`.
    pub fn start(self, software: &str, device: &str, stop: Arc<AtomicBool>) -> Result<Option<Arc<Stats>>, Error> {
        Ok(self.config(software, device, 1)?.map(|config| spawn(config, stop).1))
    }
}

/// One line for the status output of a miner.
#[must_use]
pub fn status(stats: &Stats) -> String {
    format!(
        "sky search: {} light curves in {} batches, {} refused, {} failed requests",
        stats.curves.load(Ordering::Relaxed),
        stats.batches.load(Ordering::Relaxed),
        stats.refused.load(Ordering::Relaxed),
        stats.errors.load(Ordering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;

    #[test]
    fn options_make_a_config() {
        let mut o = Options::default();
        assert!(o.clone().config("s", "d", 3).unwrap().is_none(), "without --sky there is no worker");
        assert!(o.take("--sky", "http://sky.example:8787") && o.take("--sky-address", "neb1abc"));
        assert!(!o.take("--sky-threads", "0") && !o.take("--sky-threads", "x") && !o.take("--other", "1"));
        let c = o.clone().config("neb-sky 1", "x86_64", 3).unwrap().unwrap();
        assert_eq!((c.threads, c.device.as_str(), c.address.as_str()), (3, "x86_64", "neb1abc"));
        assert!(o.take("--sky-threads", "5") && o.take("--sky-device", "card 1"));
        let c = o.config("neb-sky 1", "x86_64", 3).unwrap().unwrap();
        assert_eq!((c.threads, c.device.as_str()), (5, "card 1"));
        let mut bad = Options::default();
        assert!(bad.take("--sky", "https://sky.example"));
        assert!(bad.config("s", "d", 1).is_err());
    }

    #[test]
    fn splits_addresses() {
        assert_eq!(split("http://sky.example:8787").unwrap(), ("sky.example:8787".into(), String::new()));
        assert_eq!(split("http://sky.example/api/").unwrap(), ("sky.example:80".into(), "/api".into()));
        assert!(split("https://sky.example").is_err());
        assert!(split("http://").is_err());
        assert_eq!(encode("neb miner/1.0 \u{b5}"), "neb%20miner%2F1.0%20%C2%B5");
    }

    fn mock_server(replies: Vec<Vec<u8>>) -> (Config, JoinHandle<Vec<Vec<u8>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let config = Config {
            url: format!("http://{}", listener.local_addr().unwrap()),
            device: "card 1".into(),
            threads: 2,
            ..Config::default()
        };
        let server = thread::spawn(move || {
            let mut bodies = Vec::new();
            for reply in replies {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(&stream);
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                assert!(line.contains("device=card%201"));
                let mut length = 0;
                loop {
                    line.clear();
                    assert!(reader.read_line(&mut line).unwrap() > 0);
                    if line == "\r\n" {
                        break;
                    }
                    let (key, value) = line.split_once(':').unwrap();
                    if key.eq_ignore_ascii_case("Content-Length") {
                        length = value.trim().parse().unwrap();
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                bodies.push(body);
                stream.write_all(&reply).unwrap();
            }
            bodies
        });
        (config, server)
    }

    fn empty_response(status: u16) -> Vec<u8> {
        format!("HTTP/1.1 {status} Status\r\nConnection: close\r\nContent-Length: 0\r\n\r\n").into_bytes()
    }

    #[test]
    fn asks_searches_and_posts() {
        let mut batch = b"NEBS\x01\0\0\0\x02\0\0\0".to_vec();
        for n in [40u16, 0] {
            batch.extend_from_slice(&n.to_le_bytes());
            batch.extend((0..n).map(|i| if i % 9 < 2 { 0xB0 } else { 3 }));
        }
        let mut fixed = format!("HTTP/1.0 200 OK\r\nContent-Length: {}\r\n\r\n", batch.len()).into_bytes();
        fixed.extend_from_slice(&batch);
        let mut chunked = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n".to_vec();
        for part in batch.chunks(13) {
            write!(chunked, "{:x}\r\n", part.len()).unwrap();
            chunked.extend_from_slice(part);
            chunked.extend_from_slice(b"\r\n");
        }
        chunked.extend_from_slice(b"0\r\n\r\n");
        for reply in [fixed, chunked] {
            let (config, server) = mock_server(vec![reply, empty_response(200)]);
            let stats = Stats::default();
            assert!(round(&config, &stats).unwrap());
            assert_eq!(server.join().unwrap(), [Vec::new(), crate::run(&batch).unwrap()]);
            assert_eq!((stats.batches.load(Ordering::Relaxed), stats.curves.load(Ordering::Relaxed)), (1, 2));
        }
    }

    #[test]
    fn only_no_content_means_no_batch() {
        for status in [204, 403, 404, 429, 500] {
            let (config, server) = mock_server(vec![empty_response(status)]);
            let outcome = round(&config, &Stats::default());
            if status == 204 {
                assert!(matches!(outcome, Ok(false)));
            } else {
                assert!(matches!(outcome, Err(Error::Status(code)) if code == status));
            }
            server.join().unwrap();
        }
    }

    #[test]
    fn distinguishes_refused_results_from_a_busy_or_unavailable_server() {
        for (status, refused, failed) in [(409, 0, false), (422, 1, false), (429, 0, true), (503, 0, true)] {
            let mut batch = b"HTTP/1.0 200 OK\r\nContent-Length: 12\r\n\r\n".to_vec();
            batch.extend_from_slice(b"NEBS\x01\0\0\0\0\0\0\0");
            let (config, server) = mock_server(vec![batch, empty_response(status)]);
            let stats = Stats::default();
            let outcome = round(&config, &stats);
            if failed {
                assert!(matches!(outcome, Err(Error::Status(code)) if code == status));
            } else {
                assert!(matches!(outcome, Ok(true)));
            }
            assert_eq!(stats.refused.load(Ordering::Relaxed), refused);
            server.join().unwrap();
        }
    }
}
