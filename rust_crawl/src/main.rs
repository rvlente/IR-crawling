#![feature(map_first_last)]

mod prio_que;

use anyhow::Result;
use crossbeam::channel;
use dashmap::{DashMap, DashSet};
use parking_lot::Mutex;
use std::{
    collections::HashSet,
    io::{BufWriter, Write},
    net::IpAddr,
    path::{Path, PathBuf},
    str::FromStr,
    sync::{atomic::AtomicBool, Arc},
    time::Duration,
};
// use parking_lot::Mutex;
use prio_que::UrlPrioQue;
use rand::prelude::*;
use regex::Regex;
use select::{document::Document, predicate::Name};
use serde::{Deserialize, Serialize};
use static_init::dynamic;
use std::fmt::Debug;
use structopt::StructOpt;
use tokio::runtime::Builder;

#[dynamic]
static WEB_RE: Regex = Regex::new(r#"https?://([^/]*)/?.*"#).unwrap();

#[dynamic]
static DUTCH_URL: Regex = Regex::new(r#".*\Wnl\W.*"#).unwrap();

#[dynamic]
static RESULTS_FILE: PathBuf = Path::new("cache/results.txt").to_owned();

#[dynamic]
static NEW_LINE: Arc<str> = "\n".to_string().into();

#[dynamic]
static TAB: Arc<str> = "\t".to_string().into();

/// Extracted information from an html page
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DocData {
    text: Arc<str>,
    url: Arc<str>,
    langs: HashSet<Arc<str>>,
    links: HashSet<Arc<str>>,
}

/// Command type to control workers with
enum WorkerCmd {
    Stop,
}

enum WorkerMsg {
    Stopped,
}

struct Paths {
    que: PathBuf,
    hist: PathBuf,
    proc: PathBuf,
    dutch_w: PathBuf,
    domain_c: PathBuf,
    train_ds: PathBuf,
}

impl Paths {
    fn make(base: impl AsRef<Path>) -> Self {
        let base = base.as_ref();
        let que = base.join("que");
        let hist = base.join("hist");
        let proc = base.join("proc");
        let dutch_w = base.join("dutch_webpages");
        let domain_c = base.join("domain_counts");
        let train_ds = base.join("train_ds.jsonl");

        Self {
            que,
            hist,
            proc,
            dutch_w,
            domain_c,
            train_ds,
        }
    }
}

#[test]
fn test_paths() {
    let base = Path::new("cache");
    let paths = Paths::make(base);
    assert_eq!(paths.que, base.join("que"));
    assert_eq!(paths.hist, base.join("hist"));
    assert_eq!(paths.proc, base.join("proc"));
    assert_eq!(paths.dutch_w, base.join("dutch_webpages"));
    assert_eq!(paths.domain_c, base.join("domain_counts"));
    assert_eq!(paths.train_ds, base.join("train_ds.jsonl"));
}

trait AsRefStr {
    fn as_ref_str(&self) -> &str;
}

impl AsRefStr for String {
    fn as_ref_str(&self) -> &str {
        self.as_str()
    }
}

impl AsRefStr for &String {
    fn as_ref_str(&self) -> &str {
        self
    }
}

impl AsRefStr for Arc<str> {
    fn as_ref_str(&self) -> &str {
        self.as_ref()
    }
}

impl AsRefStr for &str {
    fn as_ref_str(&self) -> &str {
        self
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct TrainSample {
    url: Arc<str>,
    is_dutch: bool,
    #[serde(skip, default = "TrainSample::default_contents")]
    contents: Arc<str>,
}

impl TrainSample {
    fn default_contents() -> Arc<str> {
        "".to_string().into()
    }

    fn from_doc_data(doc_data: &DocData, is_dutch: bool) -> Self {
        Self {
            url: doc_data.url.clone(),
            is_dutch,
            contents: doc_data.text.clone(),
        }
    }
}

/// State of the crawler
/// - `que`: que from which workers draw websites
/// - `being processed`: websites currently being processed by workers
/// - `dutch_webpages`: websites encountered that are considered dutch
/// - `domain_counts`: amount of times a domain has been visited
#[derive(Debug, Serialize, Deserialize)]
struct CrawlerState {
    que: Mutex<UrlPrioQue>,
    history: DashSet<Arc<str>>,
    being_processed: DashSet<Arc<str>>,
    dutch_webpages: Mutex<Vec<Arc<str>>>,
    domain_counts: DashMap<Arc<str>, usize>,
    train_dataset: Mutex<Vec<TrainSample>>,
}

impl CrawlerState {
    pub fn new(start_urls: impl IntoIterator<Item = Arc<str>>) -> Self {
        Self {
            que: Mutex::new(start_urls.into_iter().map(|u| (0, 1.0, u)).collect()),
            history: DashSet::new(),
            being_processed: DashSet::new(),
            dutch_webpages: Default::default(),
            domain_counts: Default::default(),
            train_dataset: Default::default(),
        }
    }

    pub fn drain_being_processed(&mut self) {
        self.que
            .lock()
            .extend(self.being_processed.iter().map(|url| (0, 1.0, url.clone())));

        self.being_processed.clear();
    }

    fn lines_to_file(
        file: impl AsRef<Path>,
        lines: impl IntoIterator<Item = impl AsRefStr>,
        append: bool,
    ) -> Result<()> {
        let mut writer = BufWriter::new(
            std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .append(append)
                .open(file)?,
        );
        for l in lines {
            let l: &str = l.as_ref_str();
            writer.write_all(l.as_bytes())?;
        }

        Ok(())
    }

    fn url_filter(url: impl AsRefStr) -> bool {
        let url = url.as_ref_str();
        !(url.contains('\t') || url.contains('\n'))
    }

    fn domain_counts_to_lines(&self) -> impl Iterator<Item = Arc<str>> {
        let items: Vec<_> = self
            .domain_counts
            .iter()
            .map(|r| (r.key().clone(), *r.value()))
            .collect();
        items
            .into_iter()
            .filter(|(dom, _)| Self::url_filter(&**dom))
            .flat_map(|(dom, c)| [c.to_string().into(), TAB.clone(), dom, NEW_LINE.clone()])
    }

    fn drain_train_dataset_to_lines(&self) -> impl Iterator<Item = Arc<str>> {
        let items: Vec<_> = self.train_dataset.lock().drain(..).collect();
        items.into_iter().flat_map(|ts| {
            [
                serde_json::ser::to_string(&ts).unwrap().into(),
                NEW_LINE.clone(),
            ]
        })
    }

    fn que_to_lines(&self) -> impl Iterator<Item = Arc<str>> {
        let items: Vec<_> = {
            let q = self.que.lock();
            q.inner()
                .iter()
                .map(|(&c, urls)| (c, urls.clone()))
                .collect()
        };
        items
            .into_iter()
            .flat_map(|(c, us)| us.into_iter().map(|s| (c, s)).collect::<Vec<_>>())
            .filter(|(_, u)| Self::url_filter(&*u.value))
            .flat_map(|(c, u)| {
                // Some(format!("{}\t{}", c, u))
                [
                    c.to_string().into(),
                    TAB.clone(),
                    u.key.to_string().into(),
                    TAB.clone(),
                    u.value,
                    NEW_LINE.clone(),
                ]
            })
    }

    fn urls_to_lines(s: impl IntoIterator<Item = Arc<str>>) -> impl Iterator<Item = Arc<str>> {
        s.into_iter()
            .filter(|u| Self::url_filter(&**u))
            .flat_map(|url_arc| [url_arc, NEW_LINE.clone()])
        // .collect()
    }

    fn dashset_to_lines(s: &DashSet<Arc<str>>) -> impl Iterator<Item = Arc<str>> {
        let items: Vec<_> = s.iter().map(|r| r.clone()).collect();
        Self::urls_to_lines(items)
    }

    fn hist_to_lines(&self) -> impl Iterator<Item = Arc<str>> {
        Self::dashset_to_lines(&self.history)
    }

    fn being_processed_to_lines(&self) -> impl Iterator<Item = Arc<str>> {
        Self::dashset_to_lines(&self.being_processed)
    }

    fn dutch_wp_to_lines(&self) -> impl Iterator<Item = Arc<str>> {
        let items: Vec<_> = self.dutch_webpages.lock().iter().cloned().collect();
        Self::urls_to_lines(items)
    }

    fn counted_strs_from_lines(s: impl AsRef<str>) -> Vec<(usize, Arc<str>)> {
        let s = s.as_ref();
        s.lines()
            .filter_map(|l| {
                let mut split = l.split('\t');
                let count: usize = split.next()?.parse().ok()?;
                let url = split.next()?;
                Some((count, url.to_owned().into()))
            })
            .collect()
    }

    fn prio_counted_strs_from_lines(s: impl AsRef<str>) -> Vec<(usize, f32, Arc<str>)> {
        let s = s.as_ref();
        s.lines()
            .filter_map(|l| {
                let mut split = l.split('\t');
                let count: usize = split.next()?.parse().ok()?;
                let prio: f32 = split.next()?.parse().ok()?;
                let url = split.next()?;
                Some((count, prio, url.to_owned().into()))
            })
            .collect()
    }

    pub fn save_to_dir(&self, dir: impl AsRef<Path>) -> Result<()> {
        let p = dir.as_ref();
        if !p.exists() {
            std::fs::create_dir_all(p)?;
        }

        let paths = Paths::make(p);

        Self::lines_to_file(paths.que, self.que_to_lines(), false)?;
        Self::lines_to_file(paths.hist, self.hist_to_lines(), false)?;
        Self::lines_to_file(paths.proc, self.being_processed_to_lines(), false)?;
        Self::lines_to_file(paths.dutch_w, self.dutch_wp_to_lines(), false)?;
        Self::lines_to_file(paths.domain_c, self.domain_counts_to_lines(), false)?;
        Self::lines_to_file(paths.train_ds, self.drain_train_dataset_to_lines(), true)?;

        Ok(())
    }

    pub fn load_from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        use std::fs::read_to_string;
        let p = dir.as_ref();

        let paths = Paths::make(p);

        let que: UrlPrioQue = Self::prio_counted_strs_from_lines(read_to_string(paths.que)?)
            .into_iter()
            .collect();

        let history: DashSet<Arc<str>> = read_to_string(paths.hist)?
            .lines()
            .map(ToOwned::to_owned)
            .map(Arc::from)
            .collect();

        let dutch_webpages: Vec<Arc<str>> = read_to_string(paths.dutch_w)?
            .lines()
            .map(ToOwned::to_owned)
            .map(Arc::from)
            .collect();

        let being_processed: DashSet<Arc<str>> = read_to_string(paths.proc)?
            .lines()
            .map(ToOwned::to_owned)
            .map(Arc::from)
            .collect();

        let domain_counts: DashMap<Arc<str>, usize> =
            Self::counted_strs_from_lines(read_to_string(paths.domain_c)?)
                .into_iter()
                .map(|(c, u)| (u, c))
                .collect();

        Ok(Self {
            que: Mutex::new(que),
            history,
            being_processed,
            dutch_webpages: Mutex::new(dutch_webpages),
            domain_counts,
            train_dataset: Mutex::new(Vec::new()),
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
struct CrawlerConfig {
    save_file: Option<PathBuf>,
    collect_train_data: bool,
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
        start_urls: impl IntoIterator<Item = Arc<str>>,
        save_file: impl Into<Option<PathBuf>>,
        collect_train_data: bool,
    ) -> Self {
        Self {
            state: CrawlerState::new(start_urls),
            cfg: CrawlerConfig {
                save_file: save_file.into(),
                collect_train_data,
            },
        }
    }

    fn is_dutch_url(&self, url: &str) -> bool {
        if let Some(m) = WEB_RE.captures_iter(url).next() {
            let base_url = m[0].to_owned();
            DUTCH_URL.is_match(&base_url)
        } else {
            DUTCH_URL.is_match(url)
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
        url::Url::parse(url)
            .ok()
            .and_then(|u| u.host_str().map(Arc::from))
            .map(|host| self.state.domain_counts.get(&host).map(|c| *c).unwrap_or(0))
            .unwrap_or(usize::max_value())
    }

    async fn make_request(url: &str) -> Result<String> {
        // Necessary to prevent reqwest from panicking

        hyper::Uri::from_str(url)?;

        #[cfg(feature = "blocking_requests")]
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

        #[cfg(not(feature = "blocking_requests"))]
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

        let text = txt.into();
        let langs = doc
            .find(Name("html"))
            .filter_map(|n| n.attr("lang"))
            .map(ToOwned::to_owned)
            .map(Arc::from)
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
            .map(Arc::from)
            .collect();

        Ok(DocData {
            text,
            langs,
            links,
            url: url.to_owned().into(),
        })
    }

    fn increment_host_count(&self, url: Arc<str>) {
        if let Ok(url_parsed) = url::Url::parse(&*url) {
            if let Some(host) = url_parsed.host_str() {
                *self
                    .state
                    .domain_counts
                    .entry(Arc::from(host.to_string()))
                    .or_insert(0) += 1;
            }
        }
    }

    fn maybe_save_doc_as_train_sample(&self, doc: &DocData, is_dutch: bool) {
        if self.cfg.collect_train_data {
            self.state
                .train_dataset
                .lock()
                .push(TrainSample::from_doc_data(doc, is_dutch));
        }
    }

    async fn process_url(&self, url: &str) {
        if let Ok(data) = Self::get_doc_data(url).await {
            if !self.is_dutch_doc(&data) {
                if !data.langs.is_empty() {
                    self.maybe_save_doc_as_train_sample(&data, false);
                }
                return;
            }

            self.maybe_save_doc_as_train_sample(&data, true);

            println!("{}", data.url);

            self.state.dutch_webpages.lock().push(data.url);

            let filtered_urls: Vec<Arc<str>> = data
                .links
                .iter()
                // .map(|link| link.split('?').next().unwrap())
                .filter(|&link| !self.state.history.contains(link))
                .cloned()
                .collect();

            for url in filtered_urls.iter().cloned() {
                self.state.history.insert(url);
            }

            let mut rng = rand::thread_rng();

            let extension: Vec<_> = filtered_urls
                .into_iter()
                .map(|u| {
                    let hc = self.get_host_count(&u);
                    self.increment_host_count(u.clone());
                    (hc, rng.gen(), u)
                })
                .collect();

            self.state.que.lock().extend(extension);
        }
    }

    async fn worker(
        &self,
        cmd_recv: channel::Receiver<WorkerCmd>,
        msg_send: channel::Sender<WorkerMsg>,
    ) {
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
        msg_send
            .send(WorkerMsg::Stopped)
            .expect("Failed sending status to controller");
    }

    fn save(&self, save_file: impl AsRef<Path>) -> Result<()> {
        // let serialized = rmp_serde::to_vec(&self.state)?;
        let save_path = save_file.as_ref();

        if save_path.exists() {
            let mut bup_path = save_path.as_os_str().to_owned();
            bup_path.push(".old");

            let bup_path: PathBuf = bup_path.into();
            std::fs::create_dir_all(&bup_path)?;
            for entry in save_path.read_dir()? {
                let entry = entry?;
                std::fs::copy(entry.path(), bup_path.join(entry.file_name()))?;
            }
            // fs_extra::dir::r
            // fs_extra::dir::copy(save_path, bup_path, &Default::default())?;
            // std::fs::copy(&*save_file, bup_path)?;
        }

        self.state.save_to_dir(save_path)?;

        let results: String = self
            .state
            .dutch_webpages
            .lock()
            .iter()
            .map(|s| format!("{}\n", *s))
            .collect();

        // std::fs::write(&*save_file, serialized)?;
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

    fn stop_crawler(
        &self,
        to_worker: &channel::Sender<WorkerCmd>,
        from_worker: &channel::Receiver<WorkerMsg>,
        n_workers: usize,
    ) {
        eprintln!("STOPPING CRAWLER");

        for _ in 0..n_workers {
            to_worker.send(WorkerCmd::Stop).unwrap();
        }

        for _ in 0..n_workers {
            loop {
                match from_worker.recv() {
                    Ok(WorkerMsg::Stopped) => break,
                    Ok(_) => (),
                    Err(_) => {
                        eprintln!("Failed retrieving msg from worker");
                    }
                }
            }
        }
    }

    async fn run(&self, n_workers: usize) {
        let start_ip = public_ip::addr()
            .await
            .expect("Couldn't get IP address: unsafe");

        eprintln!("CURRENT IP: {:?}", start_ip);

        let running = Arc::new(AtomicBool::new(true));
        let r2 = running.clone();

        ctrlc::set_handler(move || {
            if !r2.load(std::sync::atomic::Ordering::SeqCst) {
                std::process::exit(1);
            }
            eprintln!("STOPPING CRAWLER ON NEXT LOOP, PRESS CTRL+C AGAIN TO FORCE QUIT");
            r2.store(false, std::sync::atomic::Ordering::SeqCst);
        })
        .expect("Failed setting ctlrc handler");

        async_scoped::TokioScope::scope_and_block(|scope| {
            let (to_worker, from_controller) = channel::unbounded();
            let (to_controller, from_worker) = channel::unbounded();

            for _ in 0..n_workers {
                let rcvr = from_controller.clone();
                let sndr = to_controller.clone();
                scope.spawn(self.worker(rcvr, sndr));
            }

            let mut lc: usize = 0;

            let save = || {
                eprintln!("SAVING");
                if let Err(e) = self.save_if_save_file() {
                    eprintln!("FAILED SAVING STATE: {:?}", e);
                } else {
                    eprintln!("SAVE SUCCESS");
                }
            };

            loop {
                lc += 1;
                std::thread::sleep(Duration::from_secs(30));

                if lc % 15 == 0 {
                    save();
                }

                let (ip_sndr, ip_rcvr) = channel::unbounded();
                scope.spawn(Self::send_back_public_ip(ip_sndr));
                let mut same_ip = false;

                for _ in 0..5 {
                    let cur_ip = ip_rcvr.recv_timeout(Duration::from_secs(5));

                    same_ip = if let Ok(Some(ip)) = cur_ip {
                        ip == start_ip
                    } else {
                        false
                    };

                    if same_ip {
                        break;
                    }
                }

                if self.state.being_processed.is_empty()
                    || !same_ip
                    || !running.load(std::sync::atomic::Ordering::SeqCst)
                {
                    self.stop_crawler(&to_worker, &from_worker, n_workers);
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
    #[structopt(long, help = "Collect training data for training URL classifier")]
    collect_train_data: bool,
}

fn main() {
    eprintln!("REMINDER: IS VPN ACTIVE?");
    let opt = Opt::from_args();

    let mut crawler_state: CrawlerState = opt
        .save_file
        .clone()
        .and_then(|p| CrawlerState::load_from_dir(p).ok())
        // .and_then(|state_str| {
        //     serde_json::de::from_str(&state_str)
        //         .map_err(|e| {
        //             eprintln!("{:#?}", e);
        //             e
        //         })
        //         .ok()
        // })
        .unwrap_or_else(|| {
            // panic!("WTF");
            CrawlerState::new(["https://www.wikipedia.nl/".to_string().into()])
        });

    crawler_state.drain_being_processed();

    let crawler_cfg = CrawlerConfig {
        save_file: opt.save_file.clone(),
        collect_train_data: opt.collect_train_data,
    };

    let crawler = Crawler::from((crawler_state, crawler_cfg));

    let runtime = Builder::new_multi_thread()
        .max_blocking_threads(opt.num_workers)
        // 10 mb
        .thread_stack_size(10 * 1000 * 1024)
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

#[test]
fn check_mem() {
    let sz_string = std::mem::size_of::<String>();
    let sz_arc_string = std::mem::size_of::<Arc<str>>();
    eprintln!("Stringsz: {}, Arc sz: {}", sz_string, sz_arc_string);
}

// #[test]
// fn convert_save() {
//     let original = std::fs::read_to_string("cache/state.json").unwrap();
//     let crawler: Crawler = serde_json::from_str(&original).unwrap();
//     let t = Instant::now();
//     // let state_ser = rmp_serde::to_vec(&crawler.state).unwrap();
//     // let state_ser = bincode::serialize(&crawler.state).unwrap();
//     // let state_ser = serde_pickle::to_vec(&crawler.state, Default::default()).unwrap();
//     // let state_ser = ron::to_string(&crawler.state).unwrap();
//     crawler.state.save_to_dir("cache/state").unwrap();
//     let passed = Instant::now().duration_since(t);
//     eprintln!("Saving took: {}", passed.as_millis());
//     // std::fs::write("cache/state.pkl", &state_ser).unwrap();
// }
