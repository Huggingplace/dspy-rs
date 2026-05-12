use async_trait::async_trait;
use crate::evaluate::Metric;
use crate::primitives::{Example, Module, Prediction};

/// BestOfN: runs a module N times and returns the prediction with the highest
/// metric score. Useful for stochastic sampling at temperature > 0.
pub struct BestOfN {
    module: Box<dyn Module>,
    n: usize,
    metric: Box<dyn Metric>,
}

impl BestOfN {
    pub fn new(module: impl Module + 'static, n: usize, metric: impl Metric + 'static) -> Self {
        Self {
            module: Box::new(module),
            n,
            metric: Box::new(metric),
        }
    }
}

#[async_trait]
impl Module for BestOfN {
    async fn forward(&self, input: &Example) -> anyhow::Result<Prediction> {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_prediction: Option<Prediction> = None;

        for _ in 0..self.n {
            match self.module.forward(input).await {
                Ok(prediction) => {
                    let score = self.metric.score(input, &prediction);
                    if score > best_score {
                        best_score = score;
                        best_prediction = Some(prediction);
                    }
                }
                Err(e) => {
                    tracing::warn!("BestOfN candidate failed: {e}");
                }
            }
        }

        best_prediction.ok_or_else(|| anyhow::anyhow!("All {} candidates failed", self.n))
    }
}
