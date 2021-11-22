use std::{collections::BTreeMap, iter::FromIterator, sync::Arc};

use serde::{Deserialize, Serialize};
use rand::prelude::*;
/// Priority queue optimized for many items with the same priority
/// Consits of a BTreeMap (which acts as a regular priority que)
/// with the priority as a key, and values with that priority in a
/// list associated to that key
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UrlPrioQue {
    que: BTreeMap<usize, Vec<Arc<str>>>,
}

impl UrlPrioQue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a key (domain visit count) and associated url as string into the que
    pub fn insert(&mut self, domain_count: usize, url: Arc<str>) {
        self.que
            .entry(domain_count)
            .or_insert_with(Default::default)
            .push(url);
    }

    /// Get the top element from the que, this will be a url with the minimal domain count
    pub fn pop(&mut self) -> Option<Arc<str>> {
        let first_entry = self.que.first_entry();
        if let Some(mut e) = first_entry {
            if let Some(url) = e.get_mut().pop() {
                if e.get().is_empty() {
                    self.que.pop_first();
                }
                return Some(url);
            }
        }
        None
    }

    /// Put multiple elements in the que
    pub fn extend(&mut self, items: impl IntoIterator<Item = (usize, Arc<str>)>) {
        for (c, u) in items {
            self.insert(c, u);
        }
    }

    /// Shuffle the elements of the vectors associated to the domain counts.
    pub fn shuffle_inner(&mut self) {
        self.que
            .iter_mut()
            .for_each(|(_, sites)| sites.shuffle(&mut rand::thread_rng()));
    }

    pub fn inner(&self) -> &BTreeMap<usize, Vec<Arc<str>>> {
        &self.que
    }
}

impl FromIterator<(usize, Arc<str>)> for UrlPrioQue {
    fn from_iter<T: IntoIterator<Item = (usize, Arc<str>)>>(iter: T) -> Self {
        let mut s = Self::default();
        for (c, u) in iter {
            s.insert(c, u);
        }
        s
    }
}

#[test]
fn test_url_prioque() {
    let mut q = UrlPrioQue::new();

    let insert_vals = |q: &mut UrlPrioQue| {
        for v in [5, 1, 1, 1, 7] {
            q.insert(v, v.to_string().into())
        }
    };

    insert_vals(&mut q);
    assert_eq!(q.pop(), Some("1".to_string().into()));
    assert_eq!(q.pop(), Some("1".to_string().into()));
    assert_eq!(q.pop(), Some("1".to_string().into()));
    assert_eq!(q.pop(), Some("5".to_string().into()));
    assert_eq!(q.pop(), Some("7".to_string().into()));

    insert_vals(&mut q);

    let serd = serde_json::ser::to_string(&q).unwrap();
    let deser: UrlPrioQue = serde_json::de::from_str(&serd).unwrap();

    assert_eq!(q, deser);
}
