use std::{collections::HashSet, net::IpAddr, path::{Path, PathBuf}, str::FromStr, sync::{Arc, atomic::AtomicBool}, time::Duration};
use anyhow::Result;
use rand::Rng;
use regex::Regex;
use select::{document::Document, predicate::Name};
use serde::{Deserialize, Serialize};
use data_structs::{DocData, TrainSample};
use config::{CollectTrainDataMode};
use static_init::dynamic;

use crossbeam::channel;

use self::{config::CrawlerConfig, state::CrawlerState};

pub mod config;
pub mod state;
pub mod data_structs;


#[dynamic]
static WEB_RE: Regex = Regex::new(r#"https?://([^/]*)/?.*"#).unwrap();

#[dynamic]
static DUTCH_URL: Regex = Regex::new(r#".*\Wnl\W.*"#).unwrap();


/// Command type to control workers with
enum WorkerCmd {
    Stop,
}

enum WorkerMsg {
    Stopped,
}



#[derive(Debug, Serialize, Deserialize)]
pub struct Crawler {
    pub(crate) state: CrawlerState,
    pub(crate) cfg: CrawlerConfig,
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
        collect_train_data: CollectTrainDataMode,
    ) -> Self {
        Self {
            state: CrawlerState::new(start_urls),
            cfg: CrawlerConfig {
                save_file: save_file.into(),
                collect_train_data,
                save_every: 10,
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

    async fn acquire_training_data(&self, parent_doc: &DocData) -> anyhow::Result<()> {
        //TODO fix this

        let to_store: Vec<_> = match self.cfg.collect_train_data {
            CollectTrainDataMode::Disabled => Vec::new(),
            CollectTrainDataMode::Full => {
                let mut result = Vec::new();

                for url in &parent_doc.links {
                    let doc_data = self.get_doc_data(url.as_ref()).await?;
                    let is_dutch = self.is_dutch_doc(&doc_data);

                    if is_dutch {
                        result.push(TrainSample::new(&doc_data, parent_doc.text.as_ref(), true));
                    } else if !doc_data.langs.is_empty() {
                        result.push(TrainSample::new(&doc_data, parent_doc.text.as_ref(), false));
                    }
                }

                result
            }
            CollectTrainDataMode::LinksOnly => {
                let is_dutch = self.is_dutch_doc(&parent_doc);
                if is_dutch {
                    vec![TrainSample::new(&parent_doc, "", true)]
                } else if parent_doc.langs.is_empty() {
                    vec![TrainSample::new(&parent_doc, "", false)]
                } else {
                    Vec::new()
                }
            }
        };

        if !to_store.is_empty() {
            self.state.train_dataset.lock().extend(to_store);
        }

        Ok(())
    }

    async fn get_doc_data(&self, url: &str) -> Result<DocData> {
        let txt = Self::make_request(url).await?;
        let doc = Document::from(txt.as_str());
        // let doc = Document::from_read(resp)?;

        let text: Arc<str> = txt.into();

        let langs = doc
            .find(Name("html"))
            .filter_map(|n| n.attr("lang"))
            .map(ToOwned::to_owned)
            .map(Arc::from)
            .collect();

        let links: HashSet<Arc<str>> = doc
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

    async fn process_url(&self, url: &str) {
        if let Ok(doc_data) = self.get_doc_data(url).await {
            if !self.is_dutch_doc(&doc_data) {
                return;
            }

            if self.cfg.collect_train_data != CollectTrainDataMode::Disabled {
                let _ = self.acquire_training_data(&doc_data).await;
            }

            println!("{}", doc_data.url);

            self.state.dutch_webpages.lock().push(doc_data.url);

            let filtered_urls: Vec<Arc<str>> = doc_data
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

    pub async fn run(&self, n_workers: usize) {
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

                if lc % self.cfg.save_every == 0 {
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