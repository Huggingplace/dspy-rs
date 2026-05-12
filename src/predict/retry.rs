use async_trait::async_trait;
use serde_json::Value;

use crate::adapters::{Adapter, ChatAdapter};
use crate::clients::LMConfig;
use crate::primitives::{Example, Module, Parameter, Prediction};
use crate::signatures::SignatureFields;
use crate::utils::settings;

/// Retry module: if the initial prediction fails validation, retries with
/// feedback appended to the conversation.
pub struct Retry<S: SignatureFields> {
    pub max_retries: usize,
    pub demos: Vec<Example>,
    pub config: LMConfig,
    pub validator: Option<Box<dyn Fn(&Example) -> Result<(), String> + Send + Sync>>,
    _marker: std::marker::PhantomData<S>,
}

impl<S: SignatureFields> Retry<S> {
    pub fn new() -> Self {
        Self {
            max_retries: 3,
            demos: Vec::new(),
            config: LMConfig::default(),
            validator: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_max_retries(mut self, n: usize) -> Self {
        self.max_retries = n;
        self
    }

    pub fn with_validator(
        mut self,
        f: impl Fn(&Example) -> Result<(), String> + Send + Sync + 'static,
    ) -> Self {
        self.validator = Some(Box::new(f));
        self
    }

    pub fn with_config(mut self, config: LMConfig) -> Self {
        self.config = config;
        self
    }
}

impl<S: SignatureFields> Default for Retry<S> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<S: SignatureFields + 'static> Module for Retry<S> {
    async fn forward(&self, input: &Example) -> anyhow::Result<Prediction> {
        let lm = settings::current_lm().await?;
        let adapter = ChatAdapter;
        let instruction = S::effective_instruction();
        let input_fields = S::input_fields();
        let output_fields = S::output_fields();

        let mut last_error = String::new();

        for attempt in 0..=self.max_retries {
            let effective_instruction = if attempt == 0 {
                instruction.clone()
            } else {
                format!(
                    "{}\n\nPrevious attempt was invalid: {}. Please try again.",
                    instruction, last_error
                )
            };

            let (output, _response) = adapter
                .call(
                    lm.as_ref(),
                    &self.config,
                    &effective_instruction,
                    &input_fields,
                    &output_fields,
                    &self.demos,
                    input,
                )
                .await?;

            if let Some(validator) = &self.validator {
                match validator(&output) {
                    Ok(()) => return Ok(Prediction::from_example(output)),
                    Err(e) => {
                        last_error = e;
                        continue;
                    }
                }
            } else {
                return Ok(Prediction::from_example(output));
            }
        }

        anyhow::bail!(
            "Retry exhausted {} attempts. Last error: {}",
            self.max_retries + 1,
            last_error
        )
    }

    fn named_parameters(&self) -> Vec<(&str, &dyn Parameter)> {
        vec![("retry", self as &dyn Parameter)]
    }

    fn named_parameters_mut(&mut self) -> Vec<(&str, &mut dyn Parameter)> {
        vec![("retry", self as &mut dyn Parameter)]
    }

    fn reset(&mut self) {
        self.demos.clear();
    }
}

impl<S: SignatureFields> Parameter for Retry<S> {
    fn name(&self) -> &str {
        S::signature_name()
    }

    fn dump_state(&self) -> Value {
        serde_json::json!({
            "signature": S::signature_name(),
            "module_type": "Retry",
            "max_retries": self.max_retries,
            "demos": self.demos.iter().map(|d| serde_json::to_value(d).unwrap_or_default()).collect::<Vec<_>>(),
        })
    }

    fn load_state(&mut self, state: &Value) -> anyhow::Result<()> {
        if let Some(demos) = state.get("demos").and_then(|v| v.as_array()) {
            self.demos = demos
                .iter()
                .filter_map(|d| serde_json::from_value(d.clone()).ok())
                .collect();
        }
        Ok(())
    }
}
