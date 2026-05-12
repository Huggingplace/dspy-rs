use async_trait::async_trait;
use rand::seq::SliceRandom;
use tracing::info;

use crate::evaluate::{Evaluate, Metric};
use crate::primitives::{Example, Module};

use super::Teleprompter;

/// SIMBA: Self-Improving Model-Based Assessment.
///
/// Iteratively improves a module by:
/// 1. Running the module on training examples
/// 2. Keeping successful traces as new demos
/// 3. Evaluating on a held-out dev set
/// 4. Repeating until score plateaus or max iterations reached
///
/// Unlike BootstrapFewShot which runs once, SIMBA feeds its own
/// successful outputs back as demos across multiple rounds.
pub struct SIMBA {
    pub max_iterations: usize,
    pub max_demos: usize,
    pub patience: usize,
    pub metric: std::sync::Arc<dyn Metric>,
    pub metric_threshold: Option<f64>,
    pub num_threads: usize,
}

impl SIMBA {
    pub fn new(metric: impl Metric + 'static) -> Self {
        Self {
            max_iterations: 5,
            max_demos: 8,
            patience: 2,
            metric: std::sync::Arc::new(metric),
            metric_threshold: None,
            num_threads: 4,
        }
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_max_demos(mut self, n: usize) -> Self {
        self.max_demos = n;
        self
    }

    pub fn with_patience(mut self, n: usize) -> Self {
        self.patience = n;
        self
    }

    pub fn with_metric_threshold(mut self, threshold: f64) -> Self {
        self.metric_threshold = Some(threshold);
        self
    }
}

#[async_trait]
impl Teleprompter for SIMBA {
    async fn compile(
        &self,
        module: &mut dyn Module,
        trainset: &[Example],
    ) -> anyhow::Result<()> {
        let split_point = (trainset.len() * 3) / 4;
        let (train_split, dev_set) = trainset.split_at(split_point.max(1));

        let evaluator = Evaluate::new().with_threads(self.num_threads);

        let mut best_score = f64::NEG_INFINITY;
        let mut best_state: Option<serde_json::Value> = None;
        let mut stale_rounds = 0;
        let mut accumulated_demos: Vec<Example> = Vec::new();

        for iteration in 0..self.max_iterations {
            info!(iteration, demos = accumulated_demos.len(), "SIMBA iteration");

            let mut shuffled: Vec<_> = train_split.to_vec();
            shuffled.shuffle(&mut rand::rng());

            let mut new_demos: Vec<Example> = Vec::new();

            for example in &shuffled {
                if accumulated_demos.len() + new_demos.len() >= self.max_demos {
                    break;
                }

                match module.forward(example).await {
                    Ok(prediction) => {
                        let score = self.metric.score(example, &prediction);
                        let passes = match self.metric_threshold {
                            Some(threshold) => score >= threshold,
                            None => score > 0.0,
                        };

                        if passes {
                            let mut demo = example.clone();
                            for (key, val) in prediction.completions().iter() {
                                demo.set(key, val.clone());
                            }
                            new_demos.push(demo);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("SIMBA example failed: {e}");
                    }
                }
            }

            accumulated_demos.extend(new_demos);
            if accumulated_demos.len() > self.max_demos {
                accumulated_demos.truncate(self.max_demos);
            }

            for (_name, param) in module.named_parameters_mut() {
                let state = serde_json::json!({
                    "demos": accumulated_demos.iter().map(|d| {
                        serde_json::to_value(d).unwrap_or_default()
                    }).collect::<Vec<_>>(),
                });
                param.load_state(&state)?;
            }

            let result = evaluator
                .run(module, dev_set, self.metric.as_ref())
                .await;

            info!(iteration, score = result.score, "SIMBA iteration evaluated");

            if result.score > best_score {
                best_score = result.score;
                best_state = Some(module.dump_state());
                stale_rounds = 0;
            } else {
                stale_rounds += 1;
                if stale_rounds >= self.patience {
                    info!(iteration, "SIMBA early stopping due to patience");
                    break;
                }
            }
        }

        if let Some(state) = best_state {
            module.load_state(&state)?;
        }

        info!(best_score, "SIMBA optimization complete");
        Ok(())
    }
}
