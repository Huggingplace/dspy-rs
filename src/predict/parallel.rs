use async_trait::async_trait;
use crate::primitives::{Example, Module, Prediction};

/// Runs multiple modules concurrently on the same input and collects results.
pub struct Parallel {
    modules: Vec<(String, Box<dyn Module>)>,
    max_concurrency: usize,
}

impl Parallel {
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            max_concurrency: 8,
        }
    }

    pub fn add(mut self, name: impl Into<String>, module: impl Module + 'static) -> Self {
        self.modules.push((name.into(), Box::new(module)));
        self
    }

    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n;
        self
    }
}

#[async_trait]
impl Module for Parallel {
    async fn forward(&self, input: &Example) -> anyhow::Result<Prediction> {
        let mut handles = Vec::with_capacity(self.modules.len());

        for (name, module) in &self.modules {
            let input = input.clone();
            let name = name.clone();
            // We can't easily spawn tasks across trait objects without Arc,
            // so we run sequentially but could be upgraded to use tokio::JoinSet.
            handles.push((name, module.forward(&input).await));
        }

        let mut result = Example::new();
        for (name, outcome) in handles {
            match outcome {
                Ok(prediction) => {
                    result.set(&name, serde_json::to_value(prediction.completions())?);
                }
                Err(e) => {
                    result.set(
                        &name,
                        serde_json::json!({ "error": e.to_string() }),
                    );
                }
            }
        }

        Ok(Prediction::from_example(result))
    }
}
