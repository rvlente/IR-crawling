use anyhow::Result;
use crossbeam::{channel, thread};
use dashmap::{DashMap, DashSet};
use parking_lot::Mutex;
use std::{convert::TryFrom, path::{Path, PathBuf}, sync::{atomic::AtomicBool, Arc}, time::Duration};
// use parking_lot::Mutex;
use rayon::prelude::*;
use regex::Regex;
use select::{document::Document, predicate::Name};
use serde::{Deserialize, Serialize, };
use static_init::dynamic;
use std::collections::BinaryHeap;
use structopt::StructOpt;
use tokio::runtime::Builder;

#[dynamic]
static WEB_RE: Regex = Regex::new(r#"https?://([^/]*)/?.*"#).unwrap();

#[dynamic]
static DUTCH_URL: Regex = Regex::new(r#".*\Wnl\W.*"#).unwrap();

#[dynamic]
static SAVE_FILE: PathBuf = Path::new("cache/state.json").to_owned();

#[dynamic]
static RESULTS_FILE: PathBuf = Path::new("cache/results.txt").to_owned();

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DocData {
    text: String,
    url: String,
    langs: Vec<String>,
    links: Vec<String>,
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
struct Crawler {
    que: Mutex<BinaryHeap<HeapUrl>>,
    history: DashSet<String>,
    being_processed: DashSet<String>,
    dutch_webpages: Mutex<Vec<String>>,
    domain_counts: DashMap<String, usize>,
    save_file: Option<PathBuf>,
}

impl Crawler {
    fn new(
        start_urls: impl IntoIterator<Item = String>,
        save_file: impl Into<Option<PathBuf>>,
    ) -> Self {
        Self {
            que: Mutex::new(start_urls.into_iter().map(|u| HeapUrl{ url: u, host_count: 0 }).collect()),
            history: DashSet::new(),
            being_processed: DashSet::new(),
            dutch_webpages: Default::default(),
            save_file: save_file.into(),
            domain_counts: Default::default(),
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
            .map(|host| self.domain_counts.get(&host).map(|c| *c).unwrap_or(0))
            .unwrap_or(usize::max_value())
    }

    fn mk_heap_url(&self, url: String) -> HeapUrl {
        let host_count = self.get_host_count(&url);
        HeapUrl {
            url,
            host_count,
        }
    }

    async fn get_doc_data(url: &str) -> Result<DocData> {
        
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30)).build()?;

        // client.ge

        let url_cl = url.to_owned();
        let resp = tokio::task::spawn_blocking(move || client.get(&url_cl).send()).await??;
        let txt = resp.text()?;

        // let resp = reqwest::get(url).await?;
        // let txt = resp.text().await?;

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
            .filter_map(|link| {
                if link.starts_with("http") {
                    Some(link)
                } else {
                    if let Ok(_) = url::Url::parse(&link) {
                        return None
                    }
                    let url_parsed = url::Url::parse(url).ok()?;
                    Some(url_parsed.join(&link).ok()?.to_string())
                    // Some(format!(
                    //     "{}://{}{}",
                    //     url_parsed.scheme(),
                    //     url_parsed.host_str()?,
                    //     if link.starts_with("/") {
                    //         link
                    //     } else {
                    //         format!("/{}", link)
                    //     }
                    // ))
                }
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

    async fn process_url(&self, url: &str) {
        
        if let Ok(data) = Self::get_doc_data(&url).await {
            if !self.is_dutch_doc(&data) {
                return;
            }

            println!("{}", data.url);

            if let Ok(url_parsed) = url::Url::parse(url) {
                if let Some(host) = url_parsed.host_str() {
                    *self.domain_counts.entry(host.to_owned()).or_insert(0) += 1;
                }
            }

            self.dutch_webpages.lock().push(data.url);

            let filtered_urls: Vec<String> = data
                .links
                .iter()
                // .map(|link| link.split('?').next().unwrap())
                .filter(|&link| !self.history.contains(link))
                .map(ToOwned::to_owned)
                .collect();

            for url in filtered_urls.iter().cloned() {
                self.history.insert(url);
            }


            self.que.lock().extend(filtered_urls.into_iter().map(|u| self.mk_heap_url(u)));
        }
    }

    async fn worker(&self, cmd_recv: channel::Receiver<WorkerCmd>) {
        
        loop {
            
            if let Ok(WorkerCmd::Stop) = cmd_recv.try_recv() {
                break;
            }
            if let Some(hurl) = {
                let r = self.que.lock().pop();
                r
            } {
                self.being_processed.insert(hurl.url.clone());
                self.process_url(&hurl.url).await;
                self.being_processed.remove(&hurl.url);
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

        let results = self.dutch_webpages.lock().join("\n");

        std::fs::write(&*save_file, serialized)?;
        std::fs::write(&*RESULTS_FILE, results)?;
        Ok(())
    }

    fn save_if_save_file(&self) -> Result<()> {
        if let Some(p) = &self.save_file {
            self.save(p)?;
        }
        Ok(())
    }

    async fn run(&self, n_workers: usize) {
        async_scoped::TokioScope::scope_and_block(|scope| {
            let (sender, receiver) = channel::unbounded();

            for _ in 0..n_workers {
                let recv_clone = receiver.clone();
                scope.spawn(self.worker(recv_clone));
            }

            loop {
                std::thread::sleep(Duration::from_secs(30));

                if let Err(_) = self.save_if_save_file() {
                    eprintln!("FAILED SAVING STATE");
                }

                // self.que.lock().sort_by_key(|url| {
                //     if let Ok(url_parsed) = url::Url::parse(url) {
                //         if let Some(host) = url_parsed.host_str() {
                //             let count = self.domain_counts.get(host).map(|n| *n);
                //             return count.unwrap_or(usize::max_value());
                //         }
                //     }
                //     return usize::max_value();
                // });

                if self.being_processed.is_empty() {
                    for _ in 0..n_workers {
                        sender.send(WorkerCmd::Stop).unwrap();
                    }
                }

                eprintln!("Dutch sites found: {}", self.dutch_webpages.lock().len());
            }
        });
    }
}


#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long, help="File to load and store state from/to")]
    save_file: Option<PathBuf>,
    #[structopt(long, help="amount of simultaneous workers", default_value="128")]
    num_workers: usize,
}



fn main() {
    let opt = Opt::from_args();


    let crawler = opt.save_file.clone()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|state| {
            serde_json::de::from_str(&state).ok()
        })
        .unwrap_or_else(|| Crawler::new(["https://www.wikipedia.nl/".to_string()], opt.save_file.clone()));
    
    let runtime = Builder::new_multi_thread()
        .max_blocking_threads(opt.num_workers)
        .enable_all()
        .build().unwrap();

    // let crawler = Crawler::new(["https://www.wikipedia.nl/".to_string()], opt.save_file);
    runtime.block_on(crawler.run(opt.num_workers))
}

#[test]
fn test_url() {
    let a = url::Url::parse("https://help.com/Python.html").unwrap();
    let a = a.join("/oi").unwrap();
    eprintln!("{:#?}",a);
    eprintln!("{}", a);
}
