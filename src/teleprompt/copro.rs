use async_trait::async_trait;
use tracing::info;

use crate::clients::LMConfig;
use crate::evaluate::{Evaluate, Metric};
use crate::primitives::{Example, Module};
use crate::utils::settings;

use super::Teleprompter;

/// COPRO: Collaborative Prompt Optimization.
///
/// Optimizes only the instruction (no demo selection). At each iteration,
/// proposes multiple candidate instructions and evaluates them, keeping
/// the best. Simpler than MIPROv2 but effective for instruction tuning.
pub struct COPRO {
    pub num_candidates: usize,
    pub max_iterations: usize,
    pub metric: std::sync::Arc<dyn Metric>,
    pub num_threads: usize,
    pub proposal_config: LMConfig,
}

impl COPRO {
    pub fn new(metric: impl Metric + 'static) -> Self {
        Self {
            num_candidates: 5,
            max_iterations: 3,
            metric: std::sync::Arc::new(metric),
            num_threads: 4,
            proposal_config: LMConfig {
                temperature: Some(0.9),
                max_tokens: Some(512),
                ..Default::default()
            },
        }
    }

    pub fn with_num_candidates(mut self, n: usize) -> Self {
        self.num_candidates = n;
        self
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    async fn propose_instructions(
        &self,
        current_best: &str,
        n: usize,
    ) -> Vec<String> {
        let lm = match settings::current_lm().await {
            Ok(lm) => lm,
            Err(_) => return vec![current_best.to_string()],
        };

        let prompt = format!(
            "You are optimizing a prompt instruction for a language model.\n\n\
            Current best instruction:\n\"{current_best}\"\n\n\
            Generate {n} diverse alternative instructions that might perform better. \
            Each instruction should be clear, specific, and self-contained.\n\n\
            Output each instruction on its own line, prefixed with a number and period.\n\
            Example format:\n1. <instruction>\n2. <instruction>"
        );

        let messages = vec![crate::clients::Message::user(prompt)];
        let response = match lm.complete(&messages, &self.proposal_config).await {
            Ok(r) => r,
            Err(_) => return vec![current_best.to_string()],
        };

        let mut instructions: Vec<String> = response
            .content
            .lines()
            .filter_map(|line| {
                let trimmed = line.trim();
                let stripped = trimmed
                    .trim_start_matches(|c: char| c.is_ascii_digit())
                    .trim_start_matches('.')
                    .trim_start_matches(')')
                    .trim();
                if stripped.is_empty() {
                    None
                } else {
                    Some(stripped.trim_matches('"').to_string())
                }
            })
            .collect();

        instructions.truncate(n);
        if instructions.is_empty() {
            instructions.push(current_best.to_string());
        }
        instructions
    }
}

#[async_trait]
impl Teleprompter for COPRO {
    async fn compile(
        &self,
        module: &mut dyn Module,
        trainset: &[Example],
    ) -> anyhow::Result<()> {
        let split_point = (trainset.len() * 3) / 4;
        let (_, dev_set) = trainset.split_at(split_point.max(1));

        let evaluator = Evaluate::new().with_threads(self.num_threads);

        let base_state = module.dump_state();
        let mut best_instruction = base_state
            .as_object()
            .and_then(|obj| obj.values().next())
            .and_then(|v| v.get("instruction"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let result = evaluator
            .run(module, dev_set, self.metric.as_ref())
            .await;
        let mut best_score = result.score;

        info!(best_score, "COPRO baseline score");

        for iteration in 0..self.max_iterations {
            let candidates = self
                .propose_instructions(&best_instruction, self.num_candidates)
                .await;

            for (idx, candidate) in candidates.iter().enumerate() {
                for (_name, param) in module.named_parameters_mut() {
                    let state = serde_json::json!({
                        "instruction": candidate,
                    });
                    param.load_state(&state)?;
                }

                let result = evaluator
                    .run(module, dev_set, self.metric.as_ref())
                    .await;

                info!(
                    iteration,
                    candidate_idx = idx,
                    score = result.score,
                    "COPRO candidate evaluated"
                );

                if result.score > best_score {
                    best_score = result.score;
                    best_instruction = candidate.clone();
                }
            }
        }

        for (_name, param) in module.named_parameters_mut() {
            let state = serde_json::json!({
                "instruction": best_instruction,
            });
            param.load_state(&state)?;
        }

        info!(best_score, "COPRO optimization complete");
        Ok(())
    }
}
