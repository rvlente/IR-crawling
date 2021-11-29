#![feature(map_first_last)]

mod prio_que;
mod utils;
mod crawler;

use crawler::config::CollectTrainDataMode;
use std::{path::{PathBuf} };
// use parking_lot::Mutex;
use std::fmt::Debug;
use structopt::StructOpt;
use tokio::runtime::Builder;

use crate::crawler::{Crawler, config::CrawlerConfig, state::CrawlerState};






#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long, help = "File to load and store state from/to")]
    save_file: Option<PathBuf>,
    #[structopt(long, help = "amount of simultaneous workers", default_value = "128")]
    num_workers: usize,
    // #[structopt(short, long, help = "Crawler configuration file")]
    // cfg_file: Option<PathBuf>,
    #[structopt(
        long,
        default_value = "disabled",
        help = "Mode to collect train data, must be one of: ['disabled', 'full', 'links_only']"
    )]
    collect_train_data: CollectTrainDataMode,

    #[structopt(long, help = "Save every n loops", default_value = "10")]
    save_every: usize,
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
        save_every: opt.save_every,
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
