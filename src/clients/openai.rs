use async_trait::async_trait;

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, ChatCompletionRequestAssistantMessageArgs,
        CreateChatCompletionRequestArgs,
    },
    Client,
};

use super::{LM, LMConfig, LMResponse, Message, MessageRole, Usage};

/// LM implementation backed by the `async-openai` crate.
///
/// Supports any OpenAI-compatible API (OpenAI, Azure, vLLM, Ollama, etc.)
/// by configuring the base URL.
pub struct OpenAILM {
    client: Client<OpenAIConfig>,
    model: String,
}

impl OpenAILM {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            model: model.into(),
        }
    }

    pub fn with_api_key(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        let config = OpenAIConfig::new().with_api_key(api_key);
        Self {
            client: Client::with_config(config),
            model: model.into(),
        }
    }

    pub fn with_base_url(
        model: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        let config = OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base(base_url);
        Self {
            client: Client::with_config(config),
            model: model.into(),
        }
    }

    fn to_openai_messages(
        messages: &[Message],
    ) -> Vec<ChatCompletionRequestMessage> {
        messages
            .iter()
            .map(|m| match m.role {
                MessageRole::System | MessageRole::Developer => {
                    ChatCompletionRequestSystemMessageArgs::default()
                        .content(m.content.as_str())
                        .build()
                        .unwrap()
                        .into()
                }
                MessageRole::User => {
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(m.content.as_str())
                        .build()
                        .unwrap()
                        .into()
                }
                MessageRole::Assistant => {
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .content(m.content.as_str())
                        .build()
                        .unwrap()
                        .into()
                }
            })
            .collect()
    }
}

#[async_trait]
impl LM for OpenAILM {
    async fn complete(
        &self,
        messages: &[Message],
        config: &LMConfig,
    ) -> anyhow::Result<LMResponse> {
        let openai_messages = Self::to_openai_messages(messages);

        let mut request = CreateChatCompletionRequestArgs::default();
        request.model(&self.model).messages(openai_messages);

        if let Some(temp) = config.temperature {
            request.temperature(temp as f32);
        }
        if let Some(max_tokens) = config.max_tokens {
            request.max_tokens(max_tokens as u32);
        }
        if let Some(top_p) = config.top_p {
            request.top_p(top_p as f32);
        }
        if let Some(n) = config.n {
            request.n(n as u8);
        }
        if let Some(stop) = &config.stop {
            request.stop(stop.clone());
        }

        let request = request.build()?;
        let response = self.client.chat().create(request).await?;

        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("")
            .to_string();

        let usage = response.usage.as_ref().map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        let raw = serde_json::to_value(&response).ok();

        Ok(LMResponse {
            content,
            usage,
            model: response.model,
            raw,
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}
