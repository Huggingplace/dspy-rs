use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::Stream;
use std::pin::Pin;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    #[serde(rename = "developer")]
    Developer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LMConfig {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f64>,
    pub n: Option<u32>,
    pub stop: Option<Vec<String>>,
    pub response_format: Option<Value>,
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMResponse {
    pub content: String,
    pub usage: Option<Usage>,
    pub model: String,
    pub raw: Option<Value>,
}

pub type LMStream = Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send>>;

/// The core LM abstraction — analogous to DSPy's `LM` class backed by litellm.
///
/// Implementations route to specific providers (OpenAI, Anthropic, HuggingPlace, etc.).
#[async_trait]
pub trait LM: Send + Sync {
    async fn complete(
        &self,
        messages: &[Message],
        config: &LMConfig,
    ) -> anyhow::Result<LMResponse>;

    async fn stream(
        &self,
        messages: &[Message],
        config: &LMConfig,
    ) -> anyhow::Result<LMStream> {
        let _ = (messages, config);
        anyhow::bail!("Streaming not supported by this LM implementation")
    }

    fn model_name(&self) -> &str;

    fn supports_streaming(&self) -> bool {
        false
    }
}
