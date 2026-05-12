use async_trait::async_trait;
use rand::seq::SliceRandom;
use tracing::info;

use crate::evaluate::Metric;
use crate::primitives::{Example, Module};

use super::Teleprompter;

/// Bootstrap few-shot optimizer.
///
/// Generates candidate demos by running a teacher model on training examples,
/// filtering by a metric, then assigning the best demos to the student module.
///
/// Analogous to `dspy.BootstrapFewShot`.
pub struct BootstrapFewShot {
    pub max_bootstrapped_demos: usize,
    pub max_labeled_demos: usize,
    pub max_rounds: usize,
    pub metric: Box<dyn Metric>,
    pub metric_threshold: Option<f64>,
}

impl BootstrapFewShot {
    pub fn new(metric: impl Metric + 'static) -> Self {
        Self {
            max_bootstrapped_demos: 4,
            max_labeled_demos: 16,
            max_rounds: 1,
            metric: Box::new(metric),
            metric_threshold: None,
        }
    }

    pub fn with_max_bootstrapped_demos(mut self, n: usize) -> Self {
        self.max_bootstrapped_demos = n;
        self
    }

    pub fn with_max_labeled_demos(mut self, n: usize) -> Self {
        self.max_labeled_demos = n;
        self
    }

    pub fn with_max_rounds(mut self, n: usize) -> Self {
        self.max_rounds = n;
        self
    }

    pub fn with_metric_threshold(mut self, threshold: f64) -> Self {
        self.metric_threshold = Some(threshold);
        self
    }
}

#[async_trait]
impl Teleprompter for BootstrapFewShot {
    async fn compile(
        &self,
        module: &mut dyn Module,
        trainset: &[Example],
    ) -> anyhow::Result<()> {
        let mut bootstrapped_demos: Vec<Example> = Vec::new();
        for round in 0..self.max_rounds {
            info!(round, "Starting bootstrap round");

            let mut shuffled: Vec<_> = trainset.to_vec();
            shuffled.shuffle(&mut rand::rng());

            for example in &shuffled {
                if bootstrapped_demos.len() >= self.max_bootstrapped_demos {
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
                            bootstrapped_demos.push(demo);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Bootstrap example failed: {e}");
                    }
                }
            }
        }

        let labeled_demos: Vec<Example> = trainset
            .iter()
            .take(self.max_labeled_demos)
            .cloned()
            .collect();

        let mut all_demos = bootstrapped_demos;
        all_demos.extend(labeled_demos);

        for (_name, param) in module.named_parameters_mut() {
            let state = serde_json::json!({
                "demos": all_demos.iter().map(|d| {
                    serde_json::to_value(d).unwrap_or_default()
                }).collect::<Vec<_>>(),
            });
            param.load_state(&state)?;
        }

        info!(
            total_demos = all_demos.len(),
            "Bootstrap compilation complete"
        );

        Ok(())
    }
}
