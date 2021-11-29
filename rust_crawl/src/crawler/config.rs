use std::{path::{Path, PathBuf}, str::FromStr};

use serde::{Serialize, Deserialize};
use anyhow::anyhow;

// use pyo3::prelude::*;


#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
pub enum CollectTrainDataMode {
    Disabled,
    Full,
    LinksOnly,
}

impl FromStr for CollectTrainDataMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().replace(['-', '_'], "").as_str() {
            "disabled" => Ok(Self::Disabled),
            "full" => Ok(Self::Full),
            "linksonly" => Ok(Self::LinksOnly),
            _ => Err(anyhow!("Invalid collect_train_data mode: {}", s)),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct CrawlerConfig {
    pub save_file: Option<PathBuf>,
    pub collect_train_data: CollectTrainDataMode,
    pub save_every: usize,
}

pub struct Paths {
    pub que: PathBuf,
    pub hist: PathBuf,
    pub proc: PathBuf,
    pub dutch_w: PathBuf,
    pub domain_c: PathBuf,
    pub train_ds: PathBuf,
}

impl Paths {
    pub fn make(base: impl AsRef<Path>) -> Self {
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
