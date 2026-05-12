use async_trait::async_trait;
use serde_json::Value;

use crate::adapters::{Adapter, ChatAdapter};
use crate::clients::LMConfig;
use crate::primitives::{Example, Module, Parameter, Prediction};
use crate::signatures::{FieldDescriptor, SignatureFields};
use crate::utils::settings;

/// Refine module: iteratively improves a prediction by feeding the previous
/// output back as context for the next attempt.
pub struct Refine<S: SignatureFields> {
    pub max_rounds: usize,
    pub demos: Vec<Example>,
    pub config: LMConfig,
    _marker: std::marker::PhantomData<S>,
}

impl<S: SignatureFields> Refine<S> {
    pub fn new() -> Self {
        Self {
            max_rounds: 3,
            demos: Vec::new(),
            config: LMConfig::default(),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_max_rounds(mut self, n: usize) -> Self {
        self.max_rounds = n;
        self
    }

    pub fn with_config(mut self, config: LMConfig) -> Self {
        self.config = config;
        self
    }
}

impl<S: SignatureFields> Default for Refine<S> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<S: SignatureFields + 'static> Module for Refine<S> {
    async fn forward(&self, input: &Example) -> anyhow::Result<Prediction> {
        let lm = settings::current_lm().await?;
        let adapter = ChatAdapter;
        let instruction = S::effective_instruction();

        let (mut current_output, _) = adapter
            .call(
                lm.as_ref(),
                &self.config,
                &instruction,
                &S::input_fields(),
                &S::output_fields(),
                &self.demos,
                input,
            )
            .await?;

        for _round in 1..self.max_rounds {
            let mut refined_input = input.clone();
            for (key, val) in current_output.iter() {
                refined_input.set(
                    &format!("previous_{}", key),
                    val.clone(),
                );
            }

            let refine_instruction = format!(
                "{}\n\nYou previously produced the output below. \
                 Improve it — fix errors, add detail, or make it more precise.",
                instruction
            );

            let mut extended_input_fields = S::input_fields();
            for field in S::output_fields() {
                extended_input_fields.push(FieldDescriptor {
                    name: Box::leak(format!("previous_{}", field.name).into_boxed_str()),
                    desc: Box::leak(format!("your previous {} to improve", field.name).into_boxed_str()),
                    prefix: "",
                    type_name: field.type_name,
                });
            }

            let (next_output, _) = adapter
                .call(
                    lm.as_ref(),
                    &self.config,
                    &refine_instruction,
                    &extended_input_fields,
                    &S::output_fields(),
                    &self.demos,
                    &refined_input,
                )
                .await?;

            current_output = next_output;
        }

        Ok(Prediction::from_example(current_output))
    }

    fn named_parameters(&self) -> Vec<(&str, &dyn Parameter)> {
        vec![("refine", self as &dyn Parameter)]
    }

    fn named_parameters_mut(&mut self) -> Vec<(&str, &mut dyn Parameter)> {
        vec![("refine", self as &mut dyn Parameter)]
    }

    fn reset(&mut self) {
        self.demos.clear();
    }
}

impl<S: SignatureFields> Parameter for Refine<S> {
    fn name(&self) -> &str {
        S::signature_name()
    }

    fn dump_state(&self) -> Value {
        serde_json::json!({
            "signature": S::signature_name(),
            "module_type": "Refine",
            "max_rounds": self.max_rounds,
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
