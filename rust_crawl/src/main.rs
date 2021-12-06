#![feature(map_first_last)]

mod crawler;
mod prio_que;
mod utils;

use crawler::config::CollectTrainDataMode;
use std::path::PathBuf;
// use parking_lot::Mutex;
use pyo3::prelude::*;
use std::fmt::Debug;
use structopt::StructOpt;
use tokio::runtime::Builder;

use crate::crawler::{config::CrawlerConfig, state::CrawlerState, Crawler};

// const CLASSIFIER_CODE: &str = include_str!("../../classifier_for_rust.py");

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long, help = "File to load and store state from/to")]
    save_dir: Option<PathBuf>,

    #[structopt(
        short,
        long,
        help = "Optional file to load a classifier from. If not provided, queue priority is random."
    )]
    classifier_file: Option<PathBuf>,

    #[structopt(long, help = "amount of simultaneous workers", default_value = "128")]
    num_workers: usize,

    #[structopt(
        long,
        default_value = "disabled",
        help = "Mode to collect train data, must be one of: ['disabled', 'full', 'links_only']"
    )]
    collect_train_data: CollectTrainDataMode,

    #[structopt(
        long,
        help = "Amount of characters to take as context around the urls",
        default_value = "250"
    )]
    context_size: usize,

    #[structopt(long, help = "Save every n loops", default_value = "10")]
    save_every: usize,

    // Do not worry @wikipedia, only a few articles from the seed will be downloaded, before moving on to new domains.
    #[structopt(
        long,
        help = "Seeds: URLs to start crawling from, only used if save_dir is not provided",
        default_value = "https://nl.wikipedia.org"
    )]
    seeds: Vec<String>,

    #[structopt(
        long,
        help = "Collect data for debugging/analyzing crawler performance"
    )]
    collect_debug_data: bool,
}

fn main() {
    eprintln!("REMINDER: IS VPN ACTIVE?");
    let opt = Opt::from_args();

    let mut crawler_state: CrawlerState = opt
        .save_dir
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
            if opt.seeds.is_empty() {
                panic!("No seeds nor save file provided");
            }

            CrawlerState::new(opt.seeds.iter().cloned().map(Into::into))
        });

    crawler_state.drain_being_processed();

    let crawler_cfg = CrawlerConfig {
        save_file: opt.save_dir.clone(),
        classifier_file: opt.classifier_file.clone(),
        collect_train_data: opt.collect_train_data,
        context_size: opt.context_size,
        save_every: opt.save_every,
        collect_debug_data: opt.collect_debug_data,
    };

    if let Some(_) = opt.classifier_file {
        Python::with_gil(|py| {
            let module = py.import("url_classifier").expect("Could not import python url_classifier, make sure this program is ran in a virtual environment, where the module is installed");
            let _ = module.getattr("predict_dutchiness_of_urls").expect(
                "Could not find prediction function, make sure the correct package is installed",
            );
        });
    }

    let crawler = Crawler::from((crawler_state, crawler_cfg));

    let runtime = Builder::new_multi_thread()
        .max_blocking_threads(opt.num_workers)
        // 10 mb
        .thread_stack_size(50 * 1000 * 1024)
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
    use std::sync::Arc;
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

#[test]
fn test_pyo3() {
    use pyo3::prelude::*;

    eprintln!("PYTHON STATE TESTING:");
    for _ in 0..5 {
        Python::with_gil(|py| {
            // Load directory module
            // let module = PyModule::from

            // let module = py.import("mlflow").unwrap();
            let module = py.import("url_classifier").unwrap();
            let fun = module.getattr("mutate_state").unwrap();
            let result = fun.call0().unwrap();
            eprintln!("{:#?}", result);
        });
    }

    eprintln!("=====================")

    // Python::with_gil(|py| {
    //     let result = py.eval("url_classifier. predict_dutchiness_of_urls", None, None).unwrap();
    // });
}

#[test]
fn test_pyo3_multithread() {
    use pyo3::prelude::*;

    let mut threads = Vec::new();
    for _ in 0..5 {
        let t = std::thread::spawn(|| {
            Python::with_gil(|py| {
                let module = py.import("url_classifier").unwrap();
                let fun = module.getattr("mutate_state").unwrap();
                let result = fun.call0().unwrap();
                eprintln!("{:#?}", result);
            });
        });
        threads.push(t);
    }
    for t in threads {
        t.join().unwrap();
    }
}
