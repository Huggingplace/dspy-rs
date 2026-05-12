use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

use dspy_rust::clients::{LM, LMConfig, LMResponse, Message, Usage};

/// LM implementation that routes through a HuggingPlace backend.
///
/// Calls the HuggingPlace provider adapter endpoints, which in turn route
/// to the configured provider (OpenAI, Anthropic, local runtime, etc.)
/// based on the org's routing policies.
pub struct HuggingPlaceLM {
    pub base_url: String,
    pub bearer_token: Option<String>,
    pub model: String,
    pub timeout: Duration,
    client: reqwest::Client,
}

impl HuggingPlaceLM {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            base_url: base_url.into(),
            bearer_token: None,
            model: model.into(),
            timeout: Duration::from_secs(30),
            client,
        }
    }

    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self.client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to create HTTP client");
        self
    }
}

#[async_trait]
impl LM for HuggingPlaceLM {
    async fn complete(
        &self,
        messages: &[Message],
        config: &LMConfig,
    ) -> anyhow::Result<LMResponse> {
        let endpoint = format!(
            "{}/sequence/execute",
            self.base_url.trim_end_matches('/')
        );

        let messages_json: Vec<Value> = messages
            .iter()
            .map(|m| {
                json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();

        let payload = json!({
            "model": self.model,
            "messages": messages_json,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "n": config.n.unwrap_or(1),
            "stop": config.stop,
        });

        let mut request = self.client.post(&endpoint).json(&payload);
        if let Some(token) = &self.bearer_token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("HuggingPlace API error {status}: {body}");
        }

        let body: Value = response.json().await?;

        let content = body
            .get("output")
            .or_else(|| body.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let usage = body.get("tokens").map(|t| Usage {
            prompt_tokens: t
                .get("prompt")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            completion_tokens: t
                .get("completion")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            total_tokens: t
                .get("total")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
        });

        Ok(LMResponse {
            content,
            usage,
            model: self.model.clone(),
            raw: Some(body),
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}
