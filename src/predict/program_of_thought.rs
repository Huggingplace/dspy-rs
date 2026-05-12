use async_trait::async_trait;
use serde_json::Value;

use crate::clients::{LMConfig, Message};
use crate::primitives::{Example, Module, Parameter, Prediction};
use crate::signatures::SignatureFields;
use crate::utils::settings;

use super::code_act::CodeExecutor;

/// ProgramOfThought: the LM generates a program to compute the answer,
/// which is then executed. Unlike CodeAct, there is no multi-turn loop —
/// the LM produces one program, it runs, and the output is the answer.
///
/// Analogous to `dspy.ProgramOfThought`.
pub struct ProgramOfThought<S: SignatureFields> {
    pub executor: Box<dyn CodeExecutor>,
    pub language: String,
    pub demos: Vec<Example>,
    pub config: LMConfig,
    _marker: std::marker::PhantomData<S>,
}

impl<S: SignatureFields> ProgramOfThought<S> {
    pub fn new(executor: impl CodeExecutor + 'static) -> Self {
        Self {
            executor: Box::new(executor),
            language: "python".to_string(),
            demos: Vec::new(),
            config: LMConfig::default(),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    pub fn with_config(mut self, config: LMConfig) -> Self {
        self.config = config;
        self
    }
}

#[async_trait]
impl<S: SignatureFields + 'static> Module for ProgramOfThought<S> {
    async fn forward(&self, input: &Example) -> anyhow::Result<Prediction> {
        let lm = settings::current_lm().await?;
        let instruction = S::effective_instruction();

        let output_fields: Vec<_> = S::output_fields()
            .iter()
            .map(|f| f.name.to_string())
            .collect();

        let system_prompt = format!(
            "{instruction}\n\n\
            Write a {lang} program that computes the answer. \
            The program should print the result as the last line of output.\n\
            Output fields: {fields}\n\n\
            Wrap your code in ```{lang}\n...\n``` blocks. \
            Output ONLY the code block, nothing else.",
            lang = self.language,
            fields = output_fields.join(", "),
        );

        let mut input_text = String::new();
        for field in S::input_fields() {
            if let Some(val) = input.get(field.name) {
                let text = match val {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                input_text.push_str(&format!("{}: {}\n", field.name, text));
            }
        }

        let messages = vec![
            Message::system(&system_prompt),
            Message::user(input_text),
        ];

        let response = lm.complete(&messages, &self.config).await?;
        let content = &response.content;

        let code = extract_code_block(content, &self.language)
            .ok_or_else(|| anyhow::anyhow!("No code block found in LM response"))?;

        let execution_result = self.executor.execute(&code, &self.language).await?;

        let mut result = Example::new();
        if output_fields.len() == 1 {
            result.set(
                &output_fields[0],
                Value::String(execution_result.trim().to_string()),
            );
        } else {
            match serde_json::from_str::<Value>(&execution_result) {
                Ok(Value::Object(obj)) => {
                    for field in &output_fields {
                        if let Some(val) = obj.get(field.as_str()) {
                            result.set(field, val.clone());
                        }
                    }
                }
                _ => {
                    let lines: Vec<&str> = execution_result.trim().lines().collect();
                    for (i, field) in output_fields.iter().enumerate() {
                        let val = lines.get(i).unwrap_or(&"").trim();
                        result.set(field, Value::String(val.to_string()));
                    }
                }
            }
        }

        Ok(Prediction::from_example(result))
    }

    fn named_parameters(&self) -> Vec<(&str, &dyn Parameter)> {
        vec![("program_of_thought", self as &dyn Parameter)]
    }

    fn named_parameters_mut(&mut self) -> Vec<(&str, &mut dyn Parameter)> {
        vec![("program_of_thought", self as &mut dyn Parameter)]
    }

    fn reset(&mut self) {
        self.demos.clear();
    }
}

impl<S: SignatureFields> Parameter for ProgramOfThought<S> {
    fn name(&self) -> &str {
        S::signature_name()
    }

    fn dump_state(&self) -> Value {
        serde_json::json!({
            "signature": S::signature_name(),
            "module_type": "ProgramOfThought",
            "language": self.language,
            "demos": self.demos.iter().map(|d| serde_json::to_value(d).unwrap_or_default()).collect::<Vec<_>>(),
        })
    }

    fn load_state(&mut self, state: &Value) -> anyhow::Result<()> {
        if let Some(demos) = state.get("demos").and_then(|v| v.as_array()) {
            self.demos = demos
                .iter()
                .filter_map(|d| serde_json::from_value(d.clone()).ok())
                .collect();
        }
        Ok(())
    }
}

fn extract_code_block(text: &str, language: &str) -> Option<String> {
    let fence_start = format!("```{}", language);
    if let Some(start_idx) = text.find(&fence_start) {
        let code_start = start_idx + fence_start.len();
        let rest = &text[code_start..];
        if let Some(end_idx) = rest.find("```") {
            return Some(rest[..end_idx].trim().to_string());
        }
    }
    if let Some(start_idx) = text.find("```") {
        let code_start = start_idx + 3;
        let rest = &text[code_start..];
        let rest = rest.trim_start_matches(|c: char| c.is_alphabetic() || c == '\n');
        if let Some(end_idx) = rest.find("```") {
            return Some(rest[..end_idx].trim().to_string());
        }
    }
    None
}
