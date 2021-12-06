use std::{
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
};

use super::data_structs::{DebugData, TrainSample};
use anyhow::Result;
use dashmap::{DashMap, DashSet};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use static_init::dynamic;

use crate::{prio_que::UrlPrioQue, utils::AsRefStr};

use super::config::Paths;

#[dynamic]
static NEW_LINE: Arc<str> = "\n".to_string().into();

#[dynamic]
static TAB: Arc<str> = "\t".to_string().into();

/// State of the crawler
/// - `que`: que from which workers draw websites
/// - `being processed`: websites currently being processed by workers
/// - `dutch_webpages`: websites encountered that are considered dutch
/// - `domain_counts`: amount of times a domain has been visited
#[derive(Debug, Serialize, Deserialize)]
pub struct CrawlerState {
    pub que: Mutex<UrlPrioQue>,
    pub history: DashSet<Arc<str>>,
    pub being_processed: DashSet<Arc<str>>,
    pub dutch_webpages: Mutex<Vec<Arc<str>>>,
    pub domain_counts: DashMap<Arc<str>, usize>,
    pub train_dataset: Mutex<Vec<TrainSample>>,
    pub debug_data: Mutex<Vec<DebugData>>,
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
            debug_data: Default::default(),
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

    fn items_to_json_lines<I>(items: Vec<I>) -> impl Iterator<Item = Arc<str>>
    where
        I: Serialize,
    {
        items.into_iter().flat_map(|item| {
            [
                serde_json::ser::to_string(&item).unwrap().into(),
                NEW_LINE.clone(),
            ]
        })
    }

    fn drain_train_dataset_to_lines(&self) -> impl Iterator<Item = Arc<str>> {
        let items: Vec<_> = self.train_dataset.lock().drain(..).collect();
        Self::items_to_json_lines(items)
    }

    fn drain_debug_data_to_lines(&self) -> impl Iterator<Item = Arc<str>> {
        let items: Vec<_> = self.debug_data.lock().drain(..).collect();
        Self::items_to_json_lines(items)
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
        Self::lines_to_file(paths.debug_data, self.drain_debug_data_to_lines(), true)?;

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
            debug_data: Mutex::new(Vec::new()),
        })
    }
}
