use async_trait::async_trait;
use serde_json::Value;

use super::Prediction;

/// A named, learnable parameter inside a Module (e.g., a Predict's demos or instructions).
pub trait Parameter: Send + Sync {
    fn name(&self) -> &str;
    fn dump_state(&self) -> Value;
    fn load_state(&mut self, state: &Value) -> anyhow::Result<()>;
}

/// Base trait for all DSPy modules (programs).
///
/// A Module is a composable building block. Modules contain `Parameter`s
/// (optimizable state like demos, instructions) and sub-modules. They can be
/// composed into pipelines and optimized by Teleprompters.
///
/// # Example
///
/// ```ignore
/// struct MyPipeline {
///     step1: Predict<QA>,
///     step2: ChainOfThought<Summary>,
/// }
///
/// #[async_trait]
/// impl Module for MyPipeline {
///     async fn forward(&self, input: &Example) -> anyhow::Result<Prediction> {
///         let mid = self.step1.forward(input).await?;
///         self.step2.forward(mid.completions()).await
///     }
/// }
/// ```
#[async_trait]
pub trait Module: Send + Sync {
    async fn forward(&self, input: &super::Example) -> anyhow::Result<Prediction>;

    fn named_parameters(&self) -> Vec<(&str, &dyn Parameter)> {
        Vec::new()
    }

    fn named_parameters_mut(&mut self) -> Vec<(&str, &mut dyn Parameter)> {
        Vec::new()
    }

    fn named_sub_modules(&self) -> Vec<(&str, &dyn Module)> {
        Vec::new()
    }

    fn dump_state(&self) -> Value {
        let mut state = serde_json::Map::new();
        for (name, param) in self.named_parameters() {
            state.insert(name.to_string(), param.dump_state());
        }
        Value::Object(state)
    }

    fn load_state(&mut self, state: &Value) -> anyhow::Result<()> {
        if let Some(obj) = state.as_object() {
            for (name, param) in self.named_parameters_mut() {
                if let Some(param_state) = obj.get(name) {
                    param.load_state(param_state)?;
                }
            }
        }
        Ok(())
    }

    fn reset(&mut self) {}
}
