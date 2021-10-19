use anyhow::Result;
use crossbeam::{channel, thread};
use dashmap::{DashMap, DashSet};
use parking_lot::Mutex;
use std::{path::{Path, PathBuf}, sync::{atomic::AtomicBool, Arc}, time::Duration};
// use parking_lot::Mutex;
use rayon::prelude::*;
use regex::Regex;
use select::{document::Document, predicate::Name};
use serde::{Deserialize, Serialize, __private::doc};
use static_init::dynamic;
use std::collections::BinaryHeap;


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

enum WorkerCmd {
    Stop,
}

#[derive(Debug, Serialize, Deserialize)]
struct Crawler {
    que: Mutex<Vec<String>>,
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
            que: Mutex::new(start_urls.into_iter().collect()),
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

    fn get_doc_data(url: &str) -> Result<DocData> {
        let resp = reqwest::blocking::get(url)?;
        let doc = Document::from_read(resp)?;

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
                    let url_parsed = url::Url::parse(url).ok()?;
                    Some(format!(
                        "{}://{}{}",
                        url_parsed.scheme(),
                        url_parsed.host_str()?,
                        &link
                    ))
                }
            })
            .collect();

        Ok(DocData {
            text,
            langs,
            links,
            url: url.to_owned(),
        })
    }

    fn process_url(&self, url: &str) {
        if let Ok(data) = Self::get_doc_data(&url) {

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
                .map(|link| link.split('?').next().unwrap())
                .filter(|&link| !self.history.contains(link))
                .map(ToOwned::to_owned)
                .collect();

            for url in filtered_urls.iter().cloned() {
                self.history.insert(url);
            }

            self.que.lock().extend(filtered_urls)
        }
    }

    fn worker(&self, cmd_recv: channel::Receiver<WorkerCmd>) {
        loop {
            if let Ok(WorkerCmd::Stop) = cmd_recv.try_recv() {
                break;
            }
            if let Some(url) = {
                let r = self.que.lock().pop();
                r
            } {
                self.being_processed.insert(url.clone());
                self.process_url(&url);
                self.being_processed.remove(&url);
            } else {
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }

    fn save(&self) -> Result<()> {
        let serialized = serde_json::ser::to_string(self)?;

        if SAVE_FILE.exists() {
            let mut bup_file = SAVE_FILE.as_os_str().to_owned();
            bup_file.push(".old");
            std::fs::copy(&*SAVE_FILE, bup_file)?;
        }

        let results = self.dutch_webpages.lock().join("\n");

        std::fs::write(&*SAVE_FILE, serialized)?;
        std::fs::write(&*RESULTS_FILE, results)?;
        Ok(())
    }

    fn run(&self, n_workers: usize) {
        crossbeam::scope(|scope| {
            let (sender, receiver) = channel::unbounded();

            for _ in 0..n_workers {
                let recv_clone = receiver.clone();
                scope.spawn(move |_| self.worker(recv_clone));
            }


            loop {
                std::thread::sleep(Duration::from_secs(10));

                if let Err(_) = self.save() {
                    eprintln!("FAILED SAVING");
                }

                self.que.lock().sort_by_key(|url| {
                    if let Ok(url_parsed) = url::Url::parse(url) {
                        if let Some(host) = url_parsed.host_str() {
                            let count = self.domain_counts.get(host).map(|n| *n);
                            return count.unwrap_or(usize::max_value());
                        }
                    } 
                    return usize::max_value();
                });

                if self.being_processed.is_empty() {
                    for _ in 0..n_workers {
                        sender.send(WorkerCmd::Stop).unwrap();
                    }
                }

                eprintln!("Dutch sites found: {}", self.dutch_webpages.lock().len());
            }
        })
        .unwrap();
    }
}

fn main() {
    let crawler = Crawler::new(["https://www.wikipedia.nl/".to_string()], None);
    crawler.run(100)
}

#[test]
fn test_url() {}
