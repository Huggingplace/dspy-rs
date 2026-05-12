use async_trait::async_trait;
use serde_json::Value;

use crate::adapters::{Adapter, ChatAdapter};
use crate::clients::{LMConfig, Message};
use crate::primitives::{Example, Module, Parameter, Prediction};
use crate::signatures::SignatureFields;
use crate::utils::settings;

/// A tool that can be called by the ReAct module during its action steps.
#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub desc: String,
    pub func: ToolFn,
}

type ToolFn = std::sync::Arc<dyn Fn(&str) -> String + Send + Sync>;

impl Tool {
    pub fn new(
        name: impl Into<String>,
        desc: impl Into<String>,
        func: impl Fn(&str) -> String + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            desc: desc.into(),
            func: std::sync::Arc::new(func),
        }
    }

    pub fn call(&self, input: &str) -> String {
        (self.func)(input)
    }
}

/// ReAct module: interleaves Thought/Action/Observation steps with tool calls.
///
/// At each step the LM produces a thought and an action. The action is
/// dispatched to the matching tool, and the observation is fed back.
/// This repeats until the LM produces a "Finish" action or `max_iters` is hit.
pub struct ReAct<S: SignatureFields> {
    pub tools: Vec<Tool>,
    pub max_iters: usize,
    pub demos: Vec<Example>,
    pub config: LMConfig,
    _marker: std::marker::PhantomData<S>,
}

impl<S: SignatureFields> ReAct<S> {
    pub fn new(tools: Vec<Tool>) -> Self {
        Self {
            tools,
            max_iters: 5,
            demos: Vec::new(),
            config: LMConfig::default(),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_max_iters(mut self, n: usize) -> Self {
        self.max_iters = n;
        self
    }

    pub fn with_config(mut self, config: LMConfig) -> Self {
        self.config = config;
        self
    }

    fn tool_descriptions(&self) -> String {
        self.tools
            .iter()
            .map(|t| format!("  {} - {}", t.name, t.desc))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn find_tool(&self, name: &str) -> Option<&Tool> {
        self.tools
            .iter()
            .find(|t| t.name.eq_ignore_ascii_case(name.trim()))
    }

    fn build_system_prompt(&self, instruction: &str) -> String {
        let tool_list = self.tool_descriptions();
        let output_fields: Vec<_> = S::output_fields()
            .iter()
            .map(|f| format!("`{}`", f.name))
            .collect();

        format!(
            "{instruction}\n\n\
            You have access to the following tools:\n{tool_list}\n\n\
            At each step, provide:\n\
            Thought: <your reasoning about what to do next>\n\
            Action: <tool name>\n\
            Action Input: <input to the tool>\n\n\
            When you have enough information, respond with:\n\
            Thought: I have enough information.\n\
            Action: Finish\n\
            Action Input: <not used>\n\n\
            Then provide the final output fields: {}.",
            output_fields.join(", ")
        )
    }

    fn parse_action(text: &str) -> Option<(String, String, String)> {
        let mut thought = String::new();
        let mut action = String::new();
        let mut action_input = String::new();

        for line in text.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("Thought:") {
                thought = rest.trim().to_string();
            } else if let Some(rest) = trimmed.strip_prefix("Action:") {
                action = rest.trim().to_string();
            } else if let Some(rest) = trimmed.strip_prefix("Action Input:") {
                action_input = rest.trim().to_string();
            }
        }

        if action.is_empty() {
            None
        } else {
            Some((thought, action, action_input))
        }
    }
}

#[async_trait]
impl<S: SignatureFields + 'static> Module for ReAct<S> {
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

            if let Some((_thought, action, action_input)) = Self::parse_action(content) {
                if action.eq_ignore_ascii_case("Finish") {
                    let adapter = ChatAdapter;
                    let output = adapter.parse_response(content, &S::output_fields())?;
                    return Ok(Prediction::from_example(output));
                }

                if let Some(tool) = self.find_tool(&action) {
                    let observation = tool.call(&action_input);
                    messages.push(Message::user(format!("Observation: {observation}")));
                } else {
                    messages.push(Message::user(format!(
                        "Observation: Tool '{}' not found. Available tools: {}",
                        action,
                        self.tools
                            .iter()
                            .map(|t| t.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )));
                }
            } else {
                let adapter = ChatAdapter;
                match adapter.parse_response(content, &S::output_fields()) {
                    Ok(output) => return Ok(Prediction::from_example(output)),
                    Err(_) => {
                        messages.push(Message::user(
                            "Please respond with Thought/Action/Action Input format, or provide the final output fields.".to_string(),
                        ));
                    }
                }
            }
        }

        anyhow::bail!("ReAct exceeded max iterations ({}) without producing output", self.max_iters)
    }

    fn named_parameters(&self) -> Vec<(&str, &dyn Parameter)> {
        vec![("react", self as &dyn Parameter)]
    }

    fn named_parameters_mut(&mut self) -> Vec<(&str, &mut dyn Parameter)> {
        vec![("react", self as &mut dyn Parameter)]
    }

    fn reset(&mut self) {
        self.demos.clear();
    }
}

impl<S: SignatureFields> Parameter for ReAct<S> {
    fn name(&self) -> &str {
        S::signature_name()
    }

    fn dump_state(&self) -> Value {
        serde_json::json!({
            "signature": S::signature_name(),
            "module_type": "ReAct",
            "max_iters": self.max_iters,
            "tools": self.tools.iter().map(|t| {
                serde_json::json!({ "name": t.name, "desc": t.desc })
            }).collect::<Vec<_>>(),
            "demos": self.demos.iter().map(|d| {
                serde_json::to_value(d).unwrap_or_default()
            }).collect::<Vec<_>>(),
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
