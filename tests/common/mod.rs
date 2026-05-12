use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use dspy_rust::clients::{LM, LMConfig, LMResponse, Message};
use dspy_rust::primitives::Example;
use dspy_rust::signatures::{FieldDescriptor, FromExample, SignatureFields};
use dspy_rust::Settings;
use serde_json::Value;

// ─── MockLM ───

pub struct MockLM {
    pub responses: Arc<Mutex<VecDeque<String>>>,
    pub calls: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl MockLM {
    pub fn new(responses: Vec<&str>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses.into_iter().map(String::from).collect())),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    pub fn last_messages(&self) -> Vec<Message> {
        self.calls.lock().unwrap().last().cloned().unwrap_or_default()
    }
}

#[async_trait]
impl LM for MockLM {
    async fn complete(&self, messages: &[Message], _config: &LMConfig) -> anyhow::Result<LMResponse> {
        self.calls.lock().unwrap().push(messages.to_vec());
        let content = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("MockLM: no more canned responses"))?;
        Ok(LMResponse {
            content,
            usage: None,
            model: "mock".to_string(),
            raw: None,
        })
    }

    fn model_name(&self) -> &str {
        "mock"
    }
}

// ─── MockCodeExecutor ───

pub struct MockCodeExecutor {
    pub output: String,
}

impl MockCodeExecutor {
    pub fn new(output: &str) -> Self {
        Self { output: output.to_string() }
    }
}

#[async_trait]
impl dspy_rust::CodeExecutor for MockCodeExecutor {
    async fn execute(&self, _code: &str, _language: &str) -> anyhow::Result<String> {
        Ok(self.output.clone())
    }
}

// ─── MockEmbedder ───

pub struct MockEmbedder;

#[async_trait]
impl dspy_rust::Embedder for MockEmbedder {
    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let hash = t.len() as f32;
                vec![hash / 100.0, 1.0 - hash / 100.0, 0.5]
            })
            .collect())
    }
}

// ─── Setup helper ───

pub fn setup_mock(responses: Vec<&str>) -> Arc<MockLM> {
    let mock = Arc::new(MockLM::new(responses));
    dspy_rust::configure(Settings {
        lm: Some(mock.clone()),
        ..Default::default()
    });
    mock
}

// ─── Helper to make string values less verbose ───

pub fn s(val: &str) -> Value {
    Value::String(val.to_string())
}

// ─── Test Signatures (manually implemented) ───

#[derive(serde::Serialize, serde::Deserialize)]
pub struct QA;

impl SignatureFields for QA {
    fn instruction() -> &'static str {
        "Given a question, produce a concise answer."
    }

    fn input_fields() -> Vec<FieldDescriptor> {
        vec![FieldDescriptor {
            name: "question",
            desc: "the question to answer",
            prefix: "",
            type_name: "String",
        }]
    }

    fn output_fields() -> Vec<FieldDescriptor> {
        vec![FieldDescriptor {
            name: "answer",
            desc: "a concise factual answer",
            prefix: "",
            type_name: "String",
        }]
    }

    fn signature_name() -> &'static str {
        "QA"
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Classify;

impl SignatureFields for Classify {
    fn instruction() -> &'static str {
        "Classify the text into a category."
    }

    fn input_fields() -> Vec<FieldDescriptor> {
        vec![FieldDescriptor {
            name: "text",
            desc: "text to classify",
            prefix: "",
            type_name: "String",
        }]
    }

    fn output_fields() -> Vec<FieldDescriptor> {
        vec![
            FieldDescriptor {
                name: "label",
                desc: "the category label",
                prefix: "",
                type_name: "String",
            },
            FieldDescriptor {
                name: "confidence",
                desc: "confidence score",
                prefix: "",
                type_name: "String",
            },
        ]
    }

    fn signature_name() -> &'static str {
        "Classify"
    }
}
