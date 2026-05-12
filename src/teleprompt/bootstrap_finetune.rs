use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::clients::Message;
use crate::evaluate::Metric;
use crate::primitives::{Example, Module};

use super::Teleprompter;

/// A single training example for fine-tuning, in chat format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningExample {
    pub messages: Vec<FineTuningMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningMessage {
    pub role: String,
    pub content: String,
}

impl From<&Message> for FineTuningMessage {
    fn from(m: &Message) -> Self {
        let role = match m.role {
            crate::clients::MessageRole::System => "system",
            crate::clients::MessageRole::User => "user",
            crate::clients::MessageRole::Assistant => "assistant",
            crate::clients::MessageRole::Developer => "developer",
        };
        Self {
            role: role.to_string(),
            content: m.content.clone(),
        }
    }
}

/// BootstrapFinetune: generates fine-tuning data from successful traces.
///
/// Instead of injecting demos at inference time, this optimizer:
/// 1. Runs the module on training examples
/// 2. Filters successful traces by metric
/// 3. Exports (input → output) pairs as fine-tuning data (JSONL)
///
/// The user can then use this data to fine-tune a smaller model.
/// This is a "compile to weights" strategy rather than "compile to prompts".
pub struct BootstrapFinetune {
    pub metric: Box<dyn Metric>,
    pub metric_threshold: Option<f64>,
    pub output_path: Option<String>,
}

impl BootstrapFinetune {
    pub fn new(metric: impl Metric + 'static) -> Self {
        Self {
            metric: Box::new(metric),
            metric_threshold: None,
            output_path: None,
        }
    }

    pub fn with_metric_threshold(mut self, threshold: f64) -> Self {
        self.metric_threshold = Some(threshold);
        self
    }

    pub fn with_output_path(mut self, path: impl Into<String>) -> Self {
        self.output_path = Some(path.into());
        self
    }

    pub fn generate_finetune_data(
        &self,
        traces: &[(Example, Example)],
    ) -> Vec<FineTuningExample> {
        traces
            .iter()
            .map(|(input, output)| {
                let input_text: String = input
                    .keys()
                    .map(|k| {
                        let v = input
                            .get(k)
                            .map(|v| match v {
                                serde_json::Value::String(s) => s.clone(),
                                other => other.to_string(),
                            })
                            .unwrap_or_default();
                        format!("{k}: {v}")
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                let output_text: String = output
                    .keys()
                    .map(|k| {
                        let v = output
                            .get(k)
                            .map(|v| match v {
                                serde_json::Value::String(s) => s.clone(),
                                other => other.to_string(),
                            })
                            .unwrap_or_default();
                        format!("{k}: {v}")
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                FineTuningExample {
                    messages: vec![
                        FineTuningMessage {
                            role: "user".to_string(),
                            content: input_text,
                        },
                        FineTuningMessage {
                            role: "assistant".to_string(),
                            content: output_text,
                        },
                    ],
                }
            })
            .collect()
    }
}

#[async_trait]
impl Teleprompter for BootstrapFinetune {
    async fn compile(
        &self,
        module: &mut dyn Module,
        trainset: &[Example],
    ) -> anyhow::Result<()> {
        let mut successful_traces: Vec<(Example, Example)> = Vec::new();

        for example in trainset {
            match module.forward(example).await {
                Ok(prediction) => {
                    let score = self.metric.score(example, &prediction);
                    let passes = match self.metric_threshold {
                        Some(threshold) => score >= threshold,
                        None => score > 0.0,
                    };

                    if passes {
                        let output = prediction.completions().clone();
                        successful_traces.push((example.clone(), output));
                    }
                }
                Err(e) => {
                    tracing::warn!("BootstrapFinetune example failed: {e}");
                }
            }
        }

        info!(
            total = trainset.len(),
            successful = successful_traces.len(),
            "BootstrapFinetune: traces collected"
        );

        let finetune_data = self.generate_finetune_data(&successful_traces);

        if let Some(path) = &self.output_path {
            let mut lines = Vec::new();
            for example in &finetune_data {
                lines.push(serde_json::to_string(example)?);
            }
            std::fs::write(path, lines.join("\n"))?;
            info!(path, examples = finetune_data.len(), "Fine-tuning data written");
        }

        // Also store traces as demos on the module for immediate use
        let demos: Vec<Example> = successful_traces
            .into_iter()
            .map(|(mut input, output)| {
                for (key, val) in output.iter() {
                    input.set(key, val.clone());
                }
                input
            })
            .collect();

        for (_name, param) in module.named_parameters_mut() {
            let state = serde_json::json!({
                "demos": demos.iter().map(|d| {
                    serde_json::to_value(d).unwrap_or_default()
                }).collect::<Vec<_>>(),
            });
            param.load_state(&state)?;
        }

        info!(
            demos = demos.len(),
            "BootstrapFinetune compilation complete"
        );

        Ok(())
    }
}
