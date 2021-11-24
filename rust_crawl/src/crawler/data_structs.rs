use std::{collections::HashSet, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::utils::AsRefStr;

const CONTEXT_LEN: usize = 1000;

/// Extracted information from an html page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocData {
    pub text: Arc<str>,
    pub url: Arc<str>,
    pub langs: HashSet<Arc<str>>,
    pub links: HashSet<Arc<str>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainSample {
    pub url: Arc<str>,
    pub is_dutch: bool,
    pub context: String,
    #[serde(skip, default = "TrainSample::default_contents")]
    pub contents: Arc<str>,
}

impl TrainSample {
    fn default_contents() -> Arc<str> {
        "".to_string().into()
    }

    pub fn new(doc_data: &DocData, parent_content: impl AsRefStr, is_dutch: bool) -> Self {
        let parent_content = parent_content.as_ref_str();
        let url_loc = parent_content.find(doc_data.url.as_ref_str());

        let context = if let Some(url_loc) = url_loc {
            let start = if url_loc > CONTEXT_LEN {
                url_loc - CONTEXT_LEN
            } else {
                0
            };

            let end = if url_loc + CONTEXT_LEN < parent_content.len() {
                url_loc + CONTEXT_LEN
            } else {
                parent_content.len()
            };

            parent_content.chars().take(end).skip(start).collect()
        } else {
            // std::process::exit(1);
            "".to_string()
        };

        Self {
            url: doc_data.url.clone(),
            context,
            is_dutch,
            contents: doc_data.text.clone(),
        }
    }
}