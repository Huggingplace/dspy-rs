use async_trait::async_trait;
use tracing::info;

use crate::evaluate::{Evaluate, Metric};
use crate::primitives::{Example, Module};

use super::Teleprompter;

/// BetterTogether: combines instruction optimization and demo optimization.
///
/// Runs two phases:
/// 1. Instruction optimization (using a COPRO-like approach)
/// 2. Demo bootstrapping (using BootstrapFewShot-like approach)
///
/// Then selects the combination that scores best on a dev set.
/// The insight is that good instructions and good demos are complementary.
pub struct BetterTogether {
    pub num_instruction_candidates: usize,
    pub max_bootstrapped_demos: usize,
    pub max_labeled_demos: usize,
    pub metric: std::sync::Arc<dyn Metric>,
    pub num_threads: usize,
}

impl BetterTogether {
    pub fn new(metric: impl Metric + 'static) -> Self {
        Self {
            num_instruction_candidates: 5,
            max_bootstrapped_demos: 4,
            max_labeled_demos: 16,
            metric: std::sync::Arc::new(metric),
            num_threads: 4,
        }
    }

    pub fn with_num_instruction_candidates(mut self, n: usize) -> Self {
        self.num_instruction_candidates = n;
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
}

#[async_trait]
impl Teleprompter for BetterTogether {
    async fn compile(
        &self,
        module: &mut dyn Module,
        trainset: &[Example],
    ) -> anyhow::Result<()> {
        use rand::seq::SliceRandom;

        let split_point = (trainset.len() * 3) / 4;
        let (train_split, dev_set) = trainset.split_at(split_point.max(1));

        let evaluator = Evaluate::new().with_threads(self.num_threads);
        let base_state = module.dump_state();

        // Phase 1: Instruction optimization
        // Generate candidate instructions using the LM
        let lm = crate::utils::settings::current_lm().await?;
        let base_instruction = base_state
            .as_object()
            .and_then(|obj| obj.values().next())
            .and_then(|v| v.get("instruction"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let proposal_prompt = format!(
            "You are optimizing a prompt instruction for a language model.\n\n\
            Current instruction:\n\"{base_instruction}\"\n\n\
            Generate {} diverse alternative instructions that might perform better. \
            Each should be clear and self-contained.\n\n\
            Output each on its own line, numbered.",
            self.num_instruction_candidates,
        );

        let proposal_config = crate::clients::LMConfig {
            temperature: Some(0.9),
            max_tokens: Some(512),
            ..Default::default()
        };

        let response = lm
            .complete(
                &[crate::clients::Message::user(proposal_prompt)],
                &proposal_config,
            )
            .await?;

        let mut candidate_instructions: Vec<String> = response
            .content
            .lines()
            .filter_map(|line| {
                let stripped = line
                    .trim()
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

        candidate_instructions.truncate(self.num_instruction_candidates);
        candidate_instructions.insert(0, base_instruction);

        info!(
            candidates = candidate_instructions.len(),
            "Phase 1: instruction candidates generated"
        );

        // Find best instruction
        let mut best_instruction = candidate_instructions[0].clone();
        let mut best_instruction_score = f64::NEG_INFINITY;

        for candidate in &candidate_instructions {
            module.load_state(&base_state)?;
            for (_name, param) in module.named_parameters_mut() {
                let state = serde_json::json!({ "instruction": candidate });
                param.load_state(&state)?;
            }

            let result = evaluator
                .run(module, dev_set, self.metric.as_ref())
                .await;

            if result.score > best_instruction_score {
                best_instruction_score = result.score;
                best_instruction = candidate.clone();
            }
        }

        info!(
            score = best_instruction_score,
            "Phase 1 complete: best instruction found"
        );

        // Phase 2: Demo bootstrapping with best instruction
        for (_name, param) in module.named_parameters_mut() {
            let state = serde_json::json!({ "instruction": best_instruction });
            param.load_state(&state)?;
        }

        let mut bootstrapped_demos: Vec<Example> = Vec::new();
        let mut shuffled: Vec<_> = train_split.to_vec();
        shuffled.shuffle(&mut rand::rng());

        for example in &shuffled {
            if bootstrapped_demos.len() >= self.max_bootstrapped_demos {
                break;
            }

            match module.forward(example).await {
                Ok(prediction) => {
                    let score = self.metric.score(example, &prediction);
                    if score > 0.0 {
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

        let labeled_demos: Vec<Example> = train_split
            .iter()
            .take(self.max_labeled_demos)
            .cloned()
            .collect();

        let mut all_demos = bootstrapped_demos;
        all_demos.extend(labeled_demos);

        info!(
            demos = all_demos.len(),
            "Phase 2 complete: demos bootstrapped"
        );

        // Apply best instruction + best demos
        for (_name, param) in module.named_parameters_mut() {
            let state = serde_json::json!({
                "instruction": best_instruction,
                "demos": all_demos.iter().map(|d| {
                    serde_json::to_value(d).unwrap_or_default()
                }).collect::<Vec<_>>(),
            });
            param.load_state(&state)?;
        }

        let final_result = evaluator
            .run(module, dev_set, self.metric.as_ref())
            .await;

        info!(
            final_score = final_result.score,
            "BetterTogether optimization complete"
        );

        Ok(())
    }
}
