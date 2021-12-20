use std::sync::Arc;

pub trait AsRefStr {
    fn as_ref_str(&self) -> &str;
}

impl AsRefStr for String {
    fn as_ref_str(&self) -> &str {
        self.as_str()
    }
}

impl AsRefStr for &String {
    fn as_ref_str(&self) -> &str {
        self
    }
}

impl AsRefStr for Arc<str> {
    fn as_ref_str(&self) -> &str {
        self.as_ref()
    }
}

impl AsRefStr for &str {
    fn as_ref_str(&self) -> &str {
        self
    }
}


pub fn find_seq_in_slice<T: Eq>(slice: &[T], seq: &[T]) -> Option<usize> {
    if slice.len() < seq.len() || slice.is_empty() || seq.is_empty() {
        return None;
    }

    slice.windows(seq.len()).position(|w| w == seq)
}

