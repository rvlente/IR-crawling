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
