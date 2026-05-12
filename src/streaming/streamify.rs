use async_trait::async_trait;

use crate::clients::{LMConfig, LMStream, Message};
use crate::primitives::{Example, Module, Prediction};
use crate::signatures::SignatureFields;
use crate::utils::settings;

/// A streaming wrapper around any Module.
///
/// `Streamify` wraps a signature-based module and provides a `stream_forward`
/// method that returns a token stream instead of waiting for the full response.
/// The final `forward` still returns a complete `Prediction`.
pub struct Streamify<S: SignatureFields> {
    pub config: LMConfig,
    _marker: std::marker::PhantomData<S>,
}

impl<S: SignatureFields> Streamify<S> {
    pub fn new() -> Self {
        Self {
            config: LMConfig::default(),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_config(mut self, config: LMConfig) -> Self {
        self.config = config;
        self
    }

    /// Returns a stream of token chunks for the given input.
    pub async fn stream_forward(&self, input: &Example) -> anyhow::Result<LMStream> {
        let lm = settings::current_lm().await?;
        let instruction = S::effective_instruction();

        let output_fields: Vec<_> = S::output_fields()
            .iter()
            .map(|f| f.name)
            .collect();

        let mut system_parts = vec![instruction];
        system_parts.push(format!(
            "Respond with the following fields: {}",
            output_fields.join(", ")
        ));

        let mut input_text = String::new();
        for field in S::input_fields() {
            if let Some(val) = input.get(field.name) {
                let text = match val {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                input_text.push_str(&format!("{}: {}\n", field.name, text));
            }
        }

        let messages = vec![
            Message::system(system_parts.join("\n\n")),
            Message::user(input_text),
        ];

        lm.stream(&messages, &self.config).await
    }
}

#[async_trait]
impl<S: SignatureFields + 'static> Module for Streamify<S> {
    async fn forward(&self, input: &Example) -> anyhow::Result<Prediction> {
        let lm = settings::current_lm().await?;
        let instruction = S::effective_instruction();

        let output_fields: Vec<_> = S::output_fields()
            .iter()
            .map(|f| f.name)
            .collect();

        let mut system_parts = vec![instruction];
        system_parts.push(format!(
            "Respond with the following fields: {}",
            output_fields.join(", ")
        ));

        let mut input_text = String::new();
        for field in S::input_fields() {
            if let Some(val) = input.get(field.name) {
                let text = match val {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                input_text.push_str(&format!("{}: {}\n", field.name, text));
            }
        }

        let messages = vec![
            Message::system(system_parts.join("\n\n")),
            Message::user(input_text),
        ];

        let response = lm.complete(&messages, &self.config).await?;

        let adapter = crate::adapters::ChatAdapter;
        match crate::adapters::Adapter::parse_response(
            &adapter,
            &response.content,
            &S::output_fields(),
        ) {
            Ok(output) => Ok(Prediction::from_example(output)),
            Err(_) => {
                let mut result = Example::new();
                if output_fields.len() == 1 {
                    result.set(
                        output_fields[0],
                        serde_json::Value::String(response.content.trim().to_string()),
                    );
                }
                Ok(Prediction::from_example(result))
            }
        }
    }

    fn named_parameters(&self) -> Vec<(&str, &dyn crate::primitives::Parameter)> {
        vec![]
    }

    fn named_parameters_mut(&mut self) -> Vec<(&str, &mut dyn crate::primitives::Parameter)> {
        vec![]
    }

    fn reset(&mut self) {}
}
