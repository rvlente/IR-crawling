use std::{collections::HashSet, sync::Arc};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct UrlData {
    pub url: Arc<str>,
    pub relative_url: Arc<str>,
    pub text: Arc<str>,
    pub url_context: Arc<str>,
}

/// Extracted information from an html page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocData {
    /// Natural text (no html)
    pub text: Arc<str>,
    pub url: Arc<str>,
    pub langs: HashSet<Arc<str>>,
    pub urls: HashSet<UrlData>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainSample {
    pub url: Arc<str>,
    pub is_dutch: bool,
    pub relative_url: Option<Arc<str>>,
    pub url_text: Option<Arc<str>>,
    pub url_context: Option<Arc<str>>,
}

impl TrainSample {
    pub fn new(doc_data: &DocData, url_data: impl Into<Option<UrlData>>, is_dutch: bool) -> Self {
        // let url_loc = parent_content.find(doc_data.url.as_ref_str());
        let url_data: Option<UrlData> = url_data.into();

        let (relative_url, text, parent_text) = match url_data {
            Some(UrlData {
                relative_url,
                text,
                url_context: parent_text,
                ..
            }) => (Some(relative_url), Some(text), Some(parent_text)),
            None => (None, None, None),
        };

        Self {
            url: doc_data.url.clone(),
            is_dutch,
            relative_url,
            url_text: text,
            url_context: parent_text,
        }
    }
}

// #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
// pub struct LinguaDebugData {

// }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DebugData {
    Lingua {
        predicted_dutch: bool,
        confidence: f64,
        has_dutch_lang_tag: bool,
        is_dutch_url: bool,
    },
}
