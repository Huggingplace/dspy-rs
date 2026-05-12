use indicatif::{ProgressBar, ProgressStyle};

use crate::primitives::{Example, Module};

use super::metrics::Metric;

/// Evaluates a Module on a dataset using a metric function.
///
/// Analogous to `dspy.Evaluate`. Runs the module on each example in the
/// dataset, scores with the metric, and returns aggregate results.
pub struct Evaluate {
    pub num_threads: usize,
    pub display_progress: bool,
}

#[derive(Debug, Clone)]
pub struct EvalResult {
    pub score: f64,
    pub scores: Vec<f64>,
    pub total: usize,
    pub errors: usize,
}

impl Default for Evaluate {
    fn default() -> Self {
        Self {
            num_threads: 4,
            display_progress: true,
        }
    }
}

impl Evaluate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    pub async fn run(
        &self,
        module: &dyn Module,
        dataset: &[Example],
        metric: &dyn Metric,
    ) -> EvalResult {
        let pb = if self.display_progress {
            let pb = ProgressBar::new(dataset.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };

        let mut scores = Vec::with_capacity(dataset.len());
        let mut errors = 0usize;

        for example in dataset {
            match module.forward(example).await {
                Ok(prediction) => {
                    let score = metric.score(example, &prediction);
                    scores.push(score);
                }
                Err(_) => {
                    errors += 1;
                    scores.push(0.0);
                }
            }
            if let Some(pb) = &pb {
                pb.inc(1);
            }
        }

        if let Some(pb) = pb {
            pb.finish_with_message("done");
        }

        let total = scores.len();
        let avg = if total > 0 {
            scores.iter().sum::<f64>() / total as f64
        } else {
            0.0
        };

        EvalResult {
            score: avg,
            scores,
            total,
            errors,
        }
    }
}
