use async_trait::async_trait;
use rand::seq::SliceRandom;
use tracing::info;

use crate::clients::LMConfig;
use crate::evaluate::{Evaluate, Metric};
use crate::primitives::{Example, Module};
use crate::utils::settings;

use super::Teleprompter;

/// MIPROv2: Multi-prompt Instruction Proposal Optimizer.
///
/// Jointly optimizes instructions and demo selection. At each trial:
/// 1. Propose a candidate instruction using an LM
/// 2. Select a random subset of demos
/// 3. Evaluate on a dev set
/// 4. Keep the best (instruction, demos) pair
///
/// Analogous to `dspy.MIPROv2`.
pub struct MIPROv2 {
    pub num_candidates: usize,
    pub max_bootstrapped_demos: usize,
    pub max_labeled_demos: usize,
    pub metric: std::sync::Arc<dyn Metric>,
    pub num_threads: usize,
    pub instruction_config: LMConfig,
}

impl MIPROv2 {
    pub fn new(metric: impl Metric + 'static) -> Self {
        Self {
            num_candidates: 10,
            max_bootstrapped_demos: 4,
            max_labeled_demos: 16,
            metric: std::sync::Arc::new(metric),
            num_threads: 4,
            instruction_config: LMConfig {
                temperature: Some(0.7),
                max_tokens: Some(512),
                ..Default::default()
            },
        }
    }

    pub fn with_num_candidates(mut self, n: usize) -> Self {
        self.num_candidates = n;
        self
    }

    pub fn with_max_bootstrapped_demos(mut self, n: usize) -> Self {
        self.max_bootstrapped_demos = n;
        self
    }

    async fn propose_instruction(
        &self,
        current_instruction: &str,
        input_field_names: &[String],
        output_field_names: &[String],
        demo_examples: &[Example],
    ) -> anyhow::Result<String> {
        let lm = settings::current_lm().await?;

        let demo_text = demo_examples
            .iter()
            .take(3)
            .enumerate()
            .map(|(i, d)| {
                let fields: Vec<String> = d
                    .keys()
                    .map(|k| {
                        let v = d.get(k).map(|v| match v {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        }).unwrap_or_default();
                        format!("  {}: {}", k, v)
                    })
                    .collect();
                format!("Example {}:\n{}", i + 1, fields.join("\n"))
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = format!(
            "You are an instruction optimizer for a language model pipeline.\n\n\
            The current task instruction is:\n\"{current_instruction}\"\n\n\
            Input fields: {}\n\
            Output fields: {}\n\n\
            Here are some examples from the dataset:\n{demo_text}\n\n\
            Propose an improved instruction that will help the language model \
            produce better outputs. The instruction should be clear, specific, \
            and actionable. Output ONLY the new instruction text, nothing else.",
            input_field_names.join(", "),
            output_field_names.join(", "),
        );

        let messages = vec![crate::clients::Message::user(prompt)];
        let response = lm.complete(&messages, &self.instruction_config).await?;
        Ok(response.content.trim().trim_matches('"').to_string())
    }
}

#[async_trait]
impl Teleprompter for MIPROv2 {
    async fn compile(
        &self,
        module: &mut dyn Module,
        trainset: &[Example],
    ) -> anyhow::Result<()> {
        let split_point = (trainset.len() * 3) / 4;
        let (train_split, dev_set) = trainset.split_at(split_point.max(1));

        let evaluator = Evaluate::new().with_threads(self.num_threads);

        let params = module.named_parameters();
        let base_state = module.dump_state();
        let base_instruction = base_state
            .get(params.first().map(|(n, _)| *n).unwrap_or("predict"))
            .and_then(|v| v.get("instruction"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let input_field_names: Vec<String> = base_state
            .as_object()
            .and_then(|obj| obj.values().next())
            .and_then(|v| v.get("signature"))
            .and_then(|v| v.as_str())
            .map(|_| Vec::new())
            .unwrap_or_default();

        let output_field_names: Vec<String> = Vec::new();

        let mut best_score = f64::NEG_INFINITY;
        let mut best_state: Option<serde_json::Value> = None;

        for trial in 0..self.num_candidates {
            info!(trial, "MIPROv2 trial");

            module.load_state(&base_state)?;

            let proposed_instruction = self
                .propose_instruction(
                    &base_instruction,
                    &input_field_names,
                    &output_field_names,
                    train_split,
                )
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!(trial, "Instruction proposal failed: {e}");
                    base_instruction.clone()
                });

            let mut shuffled_demos: Vec<_> = train_split.to_vec();
            shuffled_demos.shuffle(&mut rand::rng());
            let selected_demos: Vec<Example> = shuffled_demos
                .into_iter()
                .take(self.max_labeled_demos)
                .collect();

            for (_name, param) in module.named_parameters_mut() {
                let state = serde_json::json!({
                    "instruction": proposed_instruction,
                    "demos": selected_demos.iter().map(|d| {
                        serde_json::to_value(d).unwrap_or_default()
                    }).collect::<Vec<_>>(),
                });
                param.load_state(&state)?;
            }

            let result = evaluator
                .run(module, dev_set, self.metric.as_ref())
                .await;

            info!(
                trial,
                score = result.score,
                instruction_preview = &proposed_instruction[..proposed_instruction.len().min(80)],
                "Trial evaluated"
            );

            if result.score > best_score {
                best_score = result.score;
                best_state = Some(module.dump_state());
            }
        }

        if let Some(state) = best_state {
            module.load_state(&state)?;
            info!(best_score, "MIPROv2 optimization complete");
        }

        Ok(())
    }
}
