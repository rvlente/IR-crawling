#![feature(map_first_last)]

use anyhow::Result;
use crossbeam::{channel, thread};
use dashmap::{DashMap, DashSet};
use parking_lot::Mutex;
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    convert::TryFrom,
    iter::FromIterator,
    net::IpAddr,
    path::{Path, PathBuf},
    str::FromStr,
    sync::{atomic::AtomicBool, Arc},
    time::{Duration, Instant},
};
// use parking_lot::Mutex;
use rand::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use select::{document::Document, predicate::Name};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use static_init::dynamic;
use std::collections::BinaryHeap;
use std::fmt::Debug;
use structopt::StructOpt;
use tokio::runtime::Builder;

#[dynamic]
static WEB_RE: Regex = Regex::new(r#"https?://([^/]*)/?.*"#).unwrap();

#[dynamic]
static DUTCH_URL: Regex = Regex::new(r#".*\Wnl\W.*"#).unwrap();

#[dynamic]
static RESULTS_FILE: PathBuf = Path::new("cache/results.txt").to_owned();

/// Priority queue optimized for many items with the same priority
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct UrlPrioQue {
    que: BTreeMap<usize, Vec<String>>,
}

impl UrlPrioQue {
    fn new() -> Self {
        Self::default()
    }

    fn insert(&mut self, k: usize, v: String) {
        self.que
            .entry(k)
            .or_insert_with(|| Default::default())
            .push(v);
    }

    fn pop(&mut self) -> Option<String> {
        let first_entry = self.que.first_entry();
        if let Some(mut e) = first_entry {
            if let Some(url) = e.get_mut().pop() {
                if e.get().is_empty() {
                    self.que.pop_first();
                }
                return Some(url);
            }
        }
        None
    }

    fn extend(&mut self, items: impl IntoIterator<Item = (usize, String)>) {
        for (c, u) in items {
            self.insert(c, u);
        }
    }

    fn shuffle_inner(&mut self) {
        self.que
            .iter_mut()
            .for_each(|(_, sites)| sites.shuffle(&mut rand::thread_rng()));
    }
}

impl FromIterator<(usize, String)> for UrlPrioQue {
    fn from_iter<T: IntoIterator<Item = (usize, String)>>(iter: T) -> Self {
        let mut s = Self::default();
        for (c, u) in iter {
            s.insert(c, u);
        }
        s
    }
}

#[test]
fn test_url_prioque() {
    let mut q = UrlPrioQue::new();

    q.insert(5, "5".into());
    q.insert(1, "1".into());
    q.insert(1, "1".into());
    q.insert(1, "1".into());
    q.insert(7, "7".into());

    assert_eq!(q.pop(), Some("1".into()));
    assert_eq!(q.pop(), Some("1".into()));
    assert_eq!(q.pop(), Some("1".into()));
    assert_eq!(q.pop(), Some("5".into()));
    assert_eq!(q.pop(), Some("7".into()));
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DocData {
    text: String,
    url: String,
    langs: HashSet<String>,
    links: HashSet<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct HeapUrl {
    url: String,
    host_count: usize,
}

impl PartialOrd for HeapUrl {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(other.host_count.cmp(&self.host_count))
    }
}

impl Ord for HeapUrl {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

enum WorkerCmd {
    Stop,
}

#[derive(Debug, Serialize, Deserialize)]
struct CrawlerState {
    que: Mutex<UrlPrioQue>,
    history: DashSet<String>,
    being_processed: DashSet<String>,
    dutch_webpages: Mutex<Vec<String>>,
    domain_counts: DashMap<String, usize>,
}

impl CrawlerState {
    pub fn new(start_urls: impl IntoIterator<Item = String>) -> Self {
        Self {
            que: Mutex::new(start_urls.into_iter().map(|u| (0, u)).collect()),
            history: DashSet::new(),
            being_processed: DashSet::new(),
            dutch_webpages: Default::default(),
            domain_counts: Default::default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
struct CrawlerConfig {
    save_file: Option<PathBuf>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Crawler {
    state: CrawlerState,
    cfg: CrawlerConfig,
}

impl From<(CrawlerState, CrawlerConfig)> for Crawler {
    fn from((state, cfg): (CrawlerState, CrawlerConfig)) -> Self {
        Self { state, cfg }
    }
}

impl Crawler {
    fn new(
        start_urls: impl IntoIterator<Item = String>,
        save_file: impl Into<Option<PathBuf>>,
    ) -> Self {
        Self {
            state: CrawlerState::new(start_urls),
            cfg: CrawlerConfig {
                save_file: save_file.into(),
            },
        }
    }

    fn is_dutch_url(&self, url: &str) -> bool {
        if let Some(m) = WEB_RE.captures_iter(&url).next() {
            let base_url = m[0].to_owned();
            DUTCH_URL.is_match(&base_url)
        } else {
            DUTCH_URL.is_match(&url)
        }
    }

    fn is_dutch_doc(&self, doc_data: &DocData) -> bool {
        if doc_data.langs.is_empty() {
            return self.is_dutch_url(&doc_data.url);
        }

        doc_data
            .langs
            .iter()
            .any(|l| l.to_lowercase().contains("nl"))
    }

    fn get_host_count(&self, url: &str) -> usize {
        url::Url::parse(&url)
            .ok()
            .and_then(|u| u.host_str().map(ToOwned::to_owned))
            .map(|host| self.state.domain_counts.get(&host).map(|c| *c).unwrap_or(0))
            .unwrap_or(usize::max_value())
    }

    fn mk_heap_url(&self, url: String) -> HeapUrl {
        let host_count = self.get_host_count(&url);
        HeapUrl { url, host_count }
    }

    async fn make_request(url: &str) -> Result<String> {
        // Necessary to prevent reqwest from panicking

        hyper::Uri::from_str(url)?;

        #[cfg(not(feature="async_requests"))]
        let txt: Result<String> = {
            let url_cl = url.to_owned();
            tokio::task::spawn_blocking(move || {
                let client = reqwest::blocking::Client::builder()
                    .timeout(Duration::from_secs(30))
                    .build()?;
                let txt = client.get(&url_cl).send()?.text()?;
                Ok(txt)
            })
            .await?
        };

        #[cfg(feature="async_requests")]
        let txt: Result<String> = {
            let client = reqwest::Client::builder()
                    .timeout(Duration::from_secs(30))
                    .build()?;
            Ok(client.get(url).send().await?.text().await?)
        };


        txt
    }

    async fn get_doc_data(url: &str) -> Result<DocData> {
        let txt = Self::make_request(url).await?;
        let doc = Document::from(txt.as_str());
        // let doc = Document::from_read(resp)?;

        let text = "".to_string();
        let langs = doc
            .find(Name("html"))
            .filter_map(|n| n.attr("lang"))
            .map(ToOwned::to_owned)
            .collect();

        let links = doc
            .find(Name("a"))
            .filter_map(|n| n.attr("href"))
            .map(ToOwned::to_owned)
            .filter_map(|link| match url::Url::parse(&link) {
                Ok(parsed) => {
                    if parsed.scheme().starts_with("http") {
                        Some(link)
                    } else {
                        None
                    }
                }
                Err(url::ParseError::RelativeUrlWithoutBase) => {
                    let url_parsed = url::Url::parse(url).ok()?;
                    Some(url_parsed.join(&link).ok()?.to_string())
                }
                Err(_) => None,
            })
            .filter(|u| url::Url::parse(u).is_ok())
            .collect();

        Ok(DocData {
            text,
            langs,
            links,
            url: url.to_owned(),
        })
    }

    fn increment_host_count(&self, url: &str) {
        if let Ok(url_parsed) = url::Url::parse(url) {
            if let Some(host) = url_parsed.host_str() {
                *self.state.domain_counts.entry(host.to_owned()).or_insert(0) += 1;
            }
        }
    }

    async fn process_url(&self, url: &str) {
        if let Ok(data) = Self::get_doc_data(&url).await {
            if !self.is_dutch_doc(&data) {
                return;
            }

            println!("{}", data.url);

            self.state.dutch_webpages.lock().push(data.url);

            let filtered_urls: Vec<String> = data
                .links
                .iter()
                // .map(|link| link.split('?').next().unwrap())
                .filter(|&link| !self.state.history.contains(link))
                .map(ToOwned::to_owned)
                .collect();

            for url in filtered_urls.iter().cloned() {
                self.state.history.insert(url);
            }

            let extension: Vec<_> = filtered_urls
                .into_iter()
                .map(|u| {
                    let hc = self.get_host_count(&u);
                    self.increment_host_count(&u);
                    (hc, u)
                })
                .collect();

            self.state.que.lock().extend(extension);
        }
    }

    async fn worker(&self, cmd_recv: channel::Receiver<WorkerCmd>) {
        loop {
            if let Ok(WorkerCmd::Stop) = cmd_recv.try_recv() {
                break;
            }
            if let Some(hurl) = {
                let r = self.state.que.lock().pop();
                r
            } {
                self.state.being_processed.insert(hurl.clone());
                self.process_url(&hurl).await;
                self.state.being_processed.remove(&hurl);
            } else {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }

    fn save(&self, save_file: impl AsRef<Path>) -> Result<()> {
        let serialized = serde_json::ser::to_string(self)?;
        let save_file = save_file.as_ref();

        if save_file.exists() {
            let mut bup_file = save_file.as_os_str().to_owned();
            bup_file.push(".old");
            std::fs::copy(&*save_file, bup_file)?;
        }

        let results = self.state.dutch_webpages.lock().join("\n");

        std::fs::write(&*save_file, serialized)?;
        std::fs::write(&*RESULTS_FILE, results)?;
        Ok(())
    }

    fn save_if_save_file(&self) -> Result<()> {
        if let Some(p) = &self.cfg.save_file {
            self.save(p)?;
        }
        Ok(())
    }

    async fn send_back_public_ip(sender: channel::Sender<Option<IpAddr>>) {
        let _ = sender.send(public_ip::addr().await);
    }

    async fn run(&self, n_workers: usize) {
        let start_ip = public_ip::addr()
            .await
            .expect("Couldn't get IP address: unsafe");

        async_scoped::TokioScope::scope_and_block(|scope| {
            let (sender, receiver) = channel::unbounded();

            for _ in 0..n_workers {
                let recv_clone = receiver.clone();
                scope.spawn(self.worker(recv_clone));
            }

            let mut lc: usize = 0;

            let save = || {
                if let Err(_) = self.save_if_save_file() {
                    eprintln!("FAILED SAVING STATE");
                }
            };

            loop {
                lc += 1;
                std::thread::sleep(Duration::from_secs(30));

                if lc % 5 == 0 {
                    save();
                    self.state.que.lock().shuffle_inner()
                }

                if lc % 5 == 0 {}

                let (ip_sndr, ip_rcvr) = channel::unbounded();
                scope.spawn(Self::send_back_public_ip(ip_sndr));
                let cur_ip = ip_rcvr.recv_timeout(Duration::from_secs(5));

                let same_ip = if let Ok(Some(ip)) = cur_ip {
                    ip == start_ip
                } else {
                    false
                };

                if self.state.being_processed.is_empty() || !same_ip {
                    eprintln!("STOPPING CRAWLER");
                    for _ in 0..n_workers {
                        sender.send(WorkerCmd::Stop).unwrap();
                    }
                    save();
                    break;
                }

                eprintln!(
                    "Dutch sites found: {}",
                    self.state.dutch_webpages.lock().len()
                );
            }
        });
    }
}

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long, help = "File to load and store state from/to")]
    save_file: Option<PathBuf>,
    #[structopt(long, help = "amount of simultaneous workers", default_value = "128")]
    num_workers: usize,
    // #[structopt(short, long, help = "Crawler configuration file")]
    // cfg_file: Option<PathBuf>,
}

fn main() {
    let opt = Opt::from_args();

    let crawler_state: CrawlerState = opt
        .save_file
        .clone()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|state_str| serde_json::de::from_str(&state_str).ok())
        .unwrap_or_else(|| CrawlerState::new(["https://www.wikipedia.nl/".to_string()]));

    let crawler_cfg = CrawlerConfig {
        save_file: opt.save_file.clone(),
    };

    let crawler = Crawler::from((crawler_state, crawler_cfg));

    let runtime = Builder::new_multi_thread()
        .max_blocking_threads(opt.num_workers)
        .enable_all()
        .build()
        .unwrap();

    // let crawler = Crawler::new(["https://www.wikipedia.nl/".to_string()], opt.save_file);
    runtime.block_on(crawler.run(opt.num_workers));
}

#[test]
fn test_url() {
    let a = url::Url::parse("https://help.com/Python.html").unwrap();
    let a = a.join("/oi").unwrap();
    eprintln!("{:#?}", a);
    eprintln!("{}", a);
}

#[tokio::test]
async fn test_get_ip_addr() {
    // Attempt to get an IP address and print it.
    if let Some(ip) = public_ip::addr().await {
        println!("public ip address: {:?}", ip);
    } else {
        println!("couldn't get an IP address");
    }
}
