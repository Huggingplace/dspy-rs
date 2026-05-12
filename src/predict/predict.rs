use async_trait::async_trait;
use serde_json::Value;

use crate::adapters::{Adapter, ChatAdapter};
use crate::clients::{LMConfig, LMResponse};
use crate::primitives::{Example, Module, Parameter, Prediction};
use crate::signatures::SignatureFields;
use crate::utils::settings;

/// The fundamental DSPy module: takes a Signature's inputs and produces its outputs
/// by calling an LM through an Adapter.
///
/// Holds learnable state: `demos` (few-shot examples) and `instruction` override.
/// Optimizers modify these to improve performance.
pub struct Predict<S: SignatureFields> {
    pub demos: Vec<Example>,
    pub instruction: Option<String>,
    pub config: LMConfig,
    pub traces: Vec<(Example, Example)>,
    _marker: std::marker::PhantomData<S>,
}

impl<S: SignatureFields> Predict<S> {
    pub fn new() -> Self {
        Self {
            demos: Vec::new(),
            instruction: None,
            config: LMConfig::default(),
            traces: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_config(mut self, config: LMConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_demos(mut self, demos: Vec<Example>) -> Self {
        self.demos = demos;
        self
    }

    pub fn with_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = Some(instruction.into());
        self
    }

    fn effective_instruction(&self) -> String {
        self.instruction
            .clone()
            .unwrap_or_else(|| S::effective_instruction())
    }

    pub async fn call(&self, inputs: &Example) -> anyhow::Result<(Example, LMResponse)> {
        let lm = settings::current_lm().await?;
        let adapter = ChatAdapter;
        let instruction = self.effective_instruction();

        adapter
            .call(
                lm.as_ref(),
                &self.config,
                &instruction,
                &S::input_fields(),
                &S::output_fields(),
                &self.demos,
                inputs,
            )
            .await
    }
}

impl<S: SignatureFields> Default for Predict<S> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<S: SignatureFields + 'static> Module for Predict<S> {
    async fn forward(&self, input: &Example) -> anyhow::Result<Prediction> {
        let (output, _response) = self.call(input).await?;
        Ok(Prediction::from_example(output))
    }

    fn named_parameters(&self) -> Vec<(&str, &dyn Parameter)> {
        vec![("predict", self as &dyn Parameter)]
    }

    fn named_parameters_mut(&mut self) -> Vec<(&str, &mut dyn Parameter)> {
        vec![("predict", self as &mut dyn Parameter)]
    }

    fn reset(&mut self) {
        self.demos.clear();
        self.traces.clear();
        self.instruction = None;
    }
}

impl<S: SignatureFields> Parameter for Predict<S> {
    fn name(&self) -> &str {
        S::signature_name()
    }

    fn dump_state(&self) -> Value {
        serde_json::json!({
            "signature": S::signature_name(),
            "instruction": self.effective_instruction(),
            "demos": self.demos.iter().map(|d| {
                serde_json::to_value(d).unwrap_or_default()
            }).collect::<Vec<_>>(),
        })
    }

    fn load_state(&mut self, state: &Value) -> anyhow::Result<()> {
        if let Some(instruction) = state.get("instruction").and_then(|v| v.as_str()) {
            self.instruction = Some(instruction.to_string());
        }
        if let Some(demos) = state.get("demos").and_then(|v| v.as_array()) {
            self.demos = demos
                .iter()
                .filter_map(|d| serde_json::from_value(d.clone()).ok())
                .collect();
        }
        Ok(())
    }
}
