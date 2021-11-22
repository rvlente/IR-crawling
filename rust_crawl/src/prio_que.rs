use std::{
    collections::{BTreeMap, BinaryHeap},
    iter::FromIterator,
    ops::Deref,
    sync::Arc,
};

use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct CompFloat(f32);

impl CompFloat {
    pub fn new(f: f32) -> Result<Self, &'static str> {
        if f.is_finite() {
            Ok(CompFloat(f))
        } else {
            Err("Not a finite number")
        }
    }
}

impl Deref for CompFloat {
    type Target = f32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ToString for CompFloat {
    fn to_string(&self) -> String {
        self.0.to_string()
    }
}

impl Eq for CompFloat {}

impl PartialOrd for CompFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for CompFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Serialize, Deserialize)]
pub struct PrioItem {
    pub key: CompFloat,
    pub value: Arc<str>,
}

impl PrioItem {
    pub fn new(key: f32, value: Arc<str>) -> Self {
        PrioItem {
            key: CompFloat::new(key).unwrap_or_default(),
            value,
        }
    }
}

impl PartialOrd for PrioItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.key.cmp(&other.key))
    }
}

impl Ord for PrioItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Priority queue optimized for many items with the same priority
/// Consits of a BTreeMap (which acts as a regular priority que)
/// with the priority as a key, and values with that priority in a
/// list associated to that key
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct UrlPrioQue {
    que: BTreeMap<usize, BinaryHeap<PrioItem>>,
}

impl UrlPrioQue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a key (domain visit count) and associated url as string into the que
    pub fn insert(&mut self, domain_count: usize, prio: f32, url: Arc<str>) {
        self.que
            .entry(domain_count)
            .or_insert_with(Default::default)
            .push(PrioItem::new(prio, url));
    }

    /// Get the top element from the que, this will be a url with the minimal domain count
    pub fn pop(&mut self) -> Option<Arc<str>> {
        let first_entry = self.que.first_entry();
        if let Some(mut e) = first_entry {
            if let Some(url) = e.get_mut().pop() {
                if e.get().is_empty() {
                    self.que.pop_first();
                }
                return Some(url.value);
            }
        }
        None
    }

    /// Put multiple elements in the que
    pub fn extend(&mut self, items: impl IntoIterator<Item = (usize, f32, Arc<str>)>) {
        for (c, p, u) in items {
            self.insert(c, p, u);
        }
    }

    pub fn inner(&self) -> &BTreeMap<usize, BinaryHeap<PrioItem>> {
        &self.que
    }
}

impl FromIterator<(usize, f32, Arc<str>)> for UrlPrioQue {
    fn from_iter<T: IntoIterator<Item = (usize, f32, Arc<str>)>>(iter: T) -> Self {
        let mut s = Self::default();
        for (c, p, u) in iter {
            s.insert(c, p, u);
        }
        s
    }
}

#[test]
fn test_url_prioque() {
    let mut q = UrlPrioQue::new();

    let insert_vals = |q: &mut UrlPrioQue| {
        for v in [5, 1, 1, 1, 7] {
            q.insert(v, 0.0, v.to_string().into());
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

    let mut items_q = q
        .inner()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .flat_map(|(_, mut vs)| vs.drain().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut items_deser = deser
        .inner()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .flat_map(|(_, mut vs)| vs.drain().collect::<Vec<_>>())
        .collect::<Vec<_>>();

    items_q.sort_by_key(|i| i.value.clone());
    items_deser.sort_by_key(|i| i.value.clone());

    assert_eq!(items_q, items_deser);
}
