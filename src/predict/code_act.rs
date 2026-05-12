use async_trait::async_trait;
use serde_json::Value;

use crate::clients::{LMConfig, Message};
use crate::primitives::{Example, Module, Parameter, Prediction};
use crate::signatures::SignatureFields;
use crate::utils::settings;

/// Executor trait for running generated code.
#[async_trait]
pub trait CodeExecutor: Send + Sync {
    async fn execute(&self, code: &str, language: &str) -> anyhow::Result<String>;
}

/// CodeAct module: generates code to perform actions, then executes it.
///
/// The LM is prompted to produce executable code (Python, shell, etc.)
/// which is then run via the provided `CodeExecutor`. Results are fed back
/// into the conversation, and the loop continues until the LM produces
/// a final answer or hits `max_iters`.
pub struct CodeAct<S: SignatureFields> {
    pub executor: Box<dyn CodeExecutor>,
    pub language: String,
    pub max_iters: usize,
    pub demos: Vec<Example>,
    pub config: LMConfig,
    _marker: std::marker::PhantomData<S>,
}

impl<S: SignatureFields> CodeAct<S> {
    pub fn new(executor: impl CodeExecutor + 'static) -> Self {
        Self {
            executor: Box::new(executor),
            language: "python".to_string(),
            max_iters: 5,
            demos: Vec::new(),
            config: LMConfig::default(),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    pub fn with_max_iters(mut self, n: usize) -> Self {
        self.max_iters = n;
        self
    }

    pub fn with_config(mut self, config: LMConfig) -> Self {
        self.config = config;
        self
    }

    fn build_system_prompt(&self, instruction: &str) -> String {
        let output_fields: Vec<_> = S::output_fields()
            .iter()
            .map(|f| format!("`{}`", f.name))
            .collect();

        format!(
            "{instruction}\n\n\
            You can write and execute {} code to help solve this task.\n\
            Wrap code in ```{lang}\n...\n``` blocks.\n\n\
            When you have the final answer, output it clearly with the fields: {}.\n\
            Do NOT wrap the final answer in code blocks.",
            self.language,
            output_fields.join(", "),
            lang = self.language,
        )
    }

    fn extract_code(text: &str, language: &str) -> Option<String> {
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
}

#[async_trait]
impl<S: SignatureFields + 'static> Module for CodeAct<S> {
    async fn forward(&self, input: &Example) -> anyhow::Result<Prediction> {
        let lm = settings::current_lm().await?;
        let instruction = S::effective_instruction();
        let system_prompt = self.build_system_prompt(&instruction);

        let mut messages = vec![Message::system(&system_prompt)];

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
        messages.push(Message::user(input_text));

        for _ in 0..self.max_iters {
            let response = lm.complete(&messages, &self.config).await?;
            let content = &response.content;

            messages.push(Message::assistant(content.clone()));

            if let Some(code) = Self::extract_code(content, &self.language) {
                match self.executor.execute(&code, &self.language).await {
                    Ok(output) => {
                        messages.push(Message::user(format!(
                            "Code execution result:\n```\n{output}\n```"
                        )));
                    }
                    Err(e) => {
                        messages.push(Message::user(format!(
                            "Code execution error:\n```\n{e}\n```\nPlease fix the code and try again."
                        )));
                    }
                }
            } else {
                let adapter = crate::adapters::ChatAdapter;
                match crate::adapters::Adapter::parse_response(
                    &adapter,
                    content,
                    &S::output_fields(),
                ) {
                    Ok(output) => return Ok(Prediction::from_example(output)),
                    Err(_) => {
                        let mut result = Example::new();
                        if S::output_fields().len() == 1 {
                            let field = &S::output_fields()[0];
                            result.set(
                                field.name,
                                Value::String(content.trim().to_string()),
                            );
                            return Ok(Prediction::from_example(result));
                        }
                        messages.push(Message::user(
                            "Please provide the output fields or write code to compute them."
                                .to_string(),
                        ));
                    }
                }
            }
        }

        anyhow::bail!(
            "CodeAct exceeded max iterations ({}) without producing output",
            self.max_iters
        )
    }

    fn named_parameters(&self) -> Vec<(&str, &dyn Parameter)> {
        vec![("code_act", self as &dyn Parameter)]
    }

    fn named_parameters_mut(&mut self) -> Vec<(&str, &mut dyn Parameter)> {
        vec![("code_act", self as &mut dyn Parameter)]
    }

    fn reset(&mut self) {
        self.demos.clear();
    }
}

impl<S: SignatureFields> Parameter for CodeAct<S> {
    fn name(&self) -> &str {
        S::signature_name()
    }

    fn dump_state(&self) -> Value {
        serde_json::json!({
            "signature": S::signature_name(),
            "module_type": "CodeAct",
            "language": self.language,
            "max_iters": self.max_iters,
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
