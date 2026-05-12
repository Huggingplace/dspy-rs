use async_trait::async_trait;
use rand::seq::SliceRandom;
use tracing::info;

use crate::evaluate::{Evaluate, Metric};
use crate::primitives::{Example, Module};

use super::bootstrap::BootstrapFewShot;
use super::Teleprompter;

/// Runs BootstrapFewShot multiple times with different random seeds and
/// picks the demo set that scores highest on a dev set.
///
/// Analogous to `dspy.BootstrapFewShotWithRandomSearch`.
pub struct BootstrapFewShotWithRandomSearch {
    pub num_candidate_programs: usize,
    pub max_bootstrapped_demos: usize,
    pub max_labeled_demos: usize,
    pub max_rounds: usize,
    pub metric: std::sync::Arc<dyn Metric>,
    pub metric_threshold: Option<f64>,
    pub num_threads: usize,
}

impl BootstrapFewShotWithRandomSearch {
    pub fn new(metric: impl Metric + 'static) -> Self {
        Self {
            num_candidate_programs: 8,
            max_bootstrapped_demos: 4,
            max_labeled_demos: 16,
            max_rounds: 1,
            metric: std::sync::Arc::new(metric),
            metric_threshold: None,
            num_threads: 4,
        }
    }

    pub fn with_num_candidates(mut self, n: usize) -> Self {
        self.num_candidate_programs = n;
        self
    }

    pub fn with_max_bootstrapped_demos(mut self, n: usize) -> Self {
        self.max_bootstrapped_demos = n;
        self
    }

    pub fn with_max_labeled_demos(mut self, n: usize) -> Self {
        self.max_labeled_demos = n;
        self
    }

    pub fn with_metric_threshold(mut self, threshold: f64) -> Self {
        self.metric_threshold = Some(threshold);
        self
    }
}

#[async_trait]
impl Teleprompter for BootstrapFewShotWithRandomSearch {
    async fn compile(
        &self,
        module: &mut dyn Module,
        trainset: &[Example],
    ) -> anyhow::Result<()> {
        let split_point = (trainset.len() * 3) / 4;
        let (bootstrap_set, dev_set) = trainset.split_at(split_point.max(1));

        let evaluator = Evaluate::new()
            .with_threads(self.num_threads);

        let mut best_score = f64::NEG_INFINITY;
        let mut best_state: Option<serde_json::Value> = None;

        for candidate_idx in 0..self.num_candidate_programs {
            info!(candidate_idx, "Evaluating candidate program");

            module.reset();

            let mut shuffled: Vec<_> = bootstrap_set.to_vec();
            shuffled.shuffle(&mut rand::rng());

            let bootstrap = BootstrapFewShot {
                max_bootstrapped_demos: self.max_bootstrapped_demos,
                max_labeled_demos: self.max_labeled_demos,
                max_rounds: self.max_rounds,
                metric: Box::new(MetricRef(self.metric.clone())),
                metric_threshold: self.metric_threshold,
            };

            if let Err(e) = bootstrap.compile(module, &shuffled).await {
                tracing::warn!(candidate_idx, "Bootstrap failed: {e}");
                continue;
            }

            let result = evaluator
                .run(module, dev_set, self.metric.as_ref())
                .await;

            info!(
                candidate_idx,
                score = result.score,
                "Candidate evaluated"
            );

            if result.score > best_score {
                best_score = result.score;
                best_state = Some(module.dump_state());
            }
        }

        if let Some(state) = best_state {
            module.load_state(&state)?;
            info!(best_score, "Random search complete — best program loaded");
        }

        Ok(())
    }
}

struct MetricRef(std::sync::Arc<dyn Metric>);

impl Metric for MetricRef {
    fn score(
        &self,
        example: &Example,
        prediction: &crate::primitives::Prediction,
    ) -> f64 {
        self.0.score(example, prediction)
    }
}
