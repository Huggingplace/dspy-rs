use async_trait::async_trait;
use serde_json::Value;

use crate::adapters::{Adapter, ChatAdapter};
use crate::clients::LMConfig;
use crate::primitives::{Example, Module, Parameter, Prediction};
use crate::signatures::{FieldDescriptor, SignatureFields};
use crate::utils::settings;

/// Chain-of-thought module: injects a "reasoning" output field before the
/// actual output fields, prompting the LM to think step by step.
pub struct ChainOfThought<S: SignatureFields> {
    pub demos: Vec<Example>,
    pub instruction: Option<String>,
    pub config: LMConfig,
    _marker: std::marker::PhantomData<S>,
}

impl<S: SignatureFields> ChainOfThought<S> {
    pub fn new() -> Self {
        Self {
            demos: Vec::new(),
            instruction: None,
            config: LMConfig::default(),
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

    fn effective_instruction(&self) -> String {
        self.instruction
            .clone()
            .unwrap_or_else(|| S::effective_instruction())
    }

    fn output_fields_with_reasoning() -> Vec<FieldDescriptor> {
        let mut fields = vec![FieldDescriptor {
            name: "reasoning",
            desc: "Think step by step to work towards the answer.",
            prefix: "",
            type_name: "String",
        }];
        fields.extend(S::output_fields());
        fields
    }
}

impl<S: SignatureFields> Default for ChainOfThought<S> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<S: SignatureFields + 'static> Module for ChainOfThought<S> {
    async fn forward(&self, input: &Example) -> anyhow::Result<Prediction> {
        let lm = settings::current_lm().await?;
        let adapter = ChatAdapter;
        let instruction = self.effective_instruction();

        let (output, _response) = adapter
            .call(
                lm.as_ref(),
                &self.config,
                &instruction,
                &S::input_fields(),
                &Self::output_fields_with_reasoning(),
                &self.demos,
                input,
            )
            .await?;

        Ok(Prediction::from_example(output))
    }

    fn named_parameters(&self) -> Vec<(&str, &dyn Parameter)> {
        vec![("chain_of_thought", self as &dyn Parameter)]
    }

    fn named_parameters_mut(&mut self) -> Vec<(&str, &mut dyn Parameter)> {
        vec![("chain_of_thought", self as &mut dyn Parameter)]
    }

    fn reset(&mut self) {
        self.demos.clear();
        self.instruction = None;
    }
}

impl<S: SignatureFields> Parameter for ChainOfThought<S> {
    fn name(&self) -> &str {
        S::signature_name()
    }

    fn dump_state(&self) -> Value {
        serde_json::json!({
            "signature": S::signature_name(),
            "module_type": "ChainOfThought",
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
