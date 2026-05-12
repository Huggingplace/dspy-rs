use async_trait::async_trait;

use crate::primitives::{Example, Module};

use super::Teleprompter;

/// Simplest optimizer: selects `k` labeled examples as few-shot demos.
///
/// Analogous to `dspy.LabeledFewShot`. Does no bootstrapping — just picks
/// examples from the training set and assigns them as demos.
pub struct LabeledFewShot {
    pub k: usize,
}

impl LabeledFewShot {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl Default for LabeledFewShot {
    fn default() -> Self {
        Self { k: 16 }
    }
}

#[async_trait]
impl Teleprompter for LabeledFewShot {
    async fn compile(
        &self,
        module: &mut dyn Module,
        trainset: &[Example],
    ) -> anyhow::Result<()> {
        let demos: Vec<Example> = trainset.iter().take(self.k).cloned().collect();

        for (_name, param) in module.named_parameters_mut() {
            let state = serde_json::json!({
                "demos": demos.iter().map(|d| {
                    serde_json::to_value(d).unwrap_or_default()
                }).collect::<Vec<_>>(),
            });
            param.load_state(&state)?;
        }

        Ok(())
    }
}
