use async_trait::async_trait;

use crate::primitives::{Example, Module, Prediction};

/// Ensemble: runs multiple compiled modules and aggregates their outputs.
///
/// Takes multiple modules (each potentially compiled with different optimizers
/// or settings) and combines their predictions. Useful as a final step after
/// BootstrapFewShotWithRandomSearch to ensemble the top-K programs.
pub struct Ensemble {
    modules: Vec<Box<dyn Module>>,
    aggregation: AggregationStrategy,
}

pub enum AggregationStrategy {
    /// Take the first successful prediction
    First,
    /// Majority vote on a specific field
    MajorityVote(String),
}

impl Ensemble {
    pub fn new(modules: Vec<Box<dyn Module>>) -> Self {
        Self {
            modules,
            aggregation: AggregationStrategy::First,
        }
    }

    pub fn with_majority_vote(mut self, field: impl Into<String>) -> Self {
        self.aggregation = AggregationStrategy::MajorityVote(field.into());
        self
    }
}

#[async_trait]
impl Module for Ensemble {
    async fn forward(&self, input: &Example) -> anyhow::Result<Prediction> {
        let mut predictions = Vec::new();

        for module in &self.modules {
            match module.forward(input).await {
                Ok(p) => predictions.push(p),
                Err(e) => tracing::warn!("Ensemble member failed: {e}"),
            }
        }

        if predictions.is_empty() {
            anyhow::bail!("All ensemble members failed");
        }

        match &self.aggregation {
            AggregationStrategy::First => Ok(predictions.into_iter().next().unwrap()),
            AggregationStrategy::MajorityVote(field) => {
                let mut votes: std::collections::HashMap<String, usize> =
                    std::collections::HashMap::new();

                for pred in &predictions {
                    if let Some(val) = pred.get(field).and_then(|v| v.as_str()) {
                        *votes.entry(val.to_string()).or_insert(0) += 1;
                    }
                }

                let winner = votes
                    .into_iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(val, _)| val);

                if let Some(winner) = &winner {
                    let idx = predictions
                        .iter()
                        .position(|p| {
                            p.get(field).and_then(|v| v.as_str()) == Some(winner)
                        });
                    if let Some(idx) = idx {
                        return Ok(predictions.into_iter().nth(idx).unwrap());
                    }
                }

                Ok(predictions.into_iter().next().unwrap())
            }
        }
    }
}
