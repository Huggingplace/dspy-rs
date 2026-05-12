use std::sync::Arc;

use moka::future::Cache;
use serde_json::Value;

use super::{LMConfig, LMResponse, Message};

/// In-memory LM response cache backed by moka.
///
/// Keys are hashed from (model, messages, config). TTL and max capacity
/// are configurable. For persistent caching across restarts, use
/// HuggingPlace's Dragonfly/Redis layer via the integration crate.
pub struct ResponseCache {
    cache: Cache<u64, Arc<LMResponse>>,
}

impl ResponseCache {
    pub fn new(max_capacity: u64) -> Self {
        Self {
            cache: Cache::builder()
                .max_capacity(max_capacity)
                .build(),
        }
    }

    pub async fn get(
        &self,
        model: &str,
        messages: &[Message],
        config: &LMConfig,
    ) -> Option<Arc<LMResponse>> {
        let key = Self::cache_key(model, messages, config);
        self.cache.get(&key).await
    }

    pub async fn insert(
        &self,
        model: &str,
        messages: &[Message],
        config: &LMConfig,
        response: LMResponse,
    ) {
        let key = Self::cache_key(model, messages, config);
        self.cache.insert(key, Arc::new(response)).await;
    }

    fn cache_key(model: &str, messages: &[Message], config: &LMConfig) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        model.hash(&mut hasher);
        for msg in messages {
            format!("{:?}", msg.role).hash(&mut hasher);
            msg.content.hash(&mut hasher);
        }
        if let Ok(config_str) = serde_json::to_string(config) {
            config_str.hash(&mut hasher);
        }
        hasher.finish()
    }

    pub fn entry_count(&self) -> u64 {
        self.cache.entry_count()
    }

    pub async fn invalidate_all(&self) {
        self.cache.invalidate_all();
    }

    pub fn to_json_stats(&self) -> Value {
        serde_json::json!({
            "entry_count": self.entry_count(),
        })
    }
}

impl Default for ResponseCache {
    fn default() -> Self {
        Self::new(10_000)
    }
}
