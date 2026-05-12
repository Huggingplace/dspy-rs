use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::Serialize;

/// Thread-safe token usage tracker.
#[derive(Debug, Default)]
pub struct UsageTracker {
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    requests: AtomicU64,
}

#[derive(Debug, Clone, Serialize)]
pub struct UsageSummary {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub requests: u64,
}

impl UsageTracker {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn record(&self, prompt_tokens: u32, completion_tokens: u32) {
        self.prompt_tokens
            .fetch_add(prompt_tokens as u64, Ordering::Relaxed);
        self.completion_tokens
            .fetch_add(completion_tokens as u64, Ordering::Relaxed);
        self.requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn summary(&self) -> UsageSummary {
        let prompt = self.prompt_tokens.load(Ordering::Relaxed);
        let completion = self.completion_tokens.load(Ordering::Relaxed);
        UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            requests: self.requests.load(Ordering::Relaxed),
        }
    }

    pub fn reset(&self) {
        self.prompt_tokens.store(0, Ordering::Relaxed);
        self.completion_tokens.store(0, Ordering::Relaxed);
        self.requests.store(0, Ordering::Relaxed);
    }
}
