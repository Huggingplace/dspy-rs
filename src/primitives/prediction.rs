use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::Example;

/// Wraps an `Example` with additional metadata about how the prediction was produced.
///
/// Analogous to DSPy's `Prediction` — stores the LM output fields plus
/// optional trace information for optimizer introspection.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Prediction {
    #[serde(flatten)]
    pub example: Example,
    #[serde(skip)]
    pub trace: Option<Vec<TraceEntry>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    pub module_name: String,
    pub input: Example,
    pub output: Example,
}

impl Prediction {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_example(example: Example) -> Self {
        Self {
            example,
            trace: None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.example.get(key)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.example.get_str(key)
    }

    pub fn set(&mut self, key: &str, value: Value) {
        self.example.set(key, value);
    }

    pub fn with_trace(mut self, trace: Vec<TraceEntry>) -> Self {
        self.trace = Some(trace);
        self
    }

    pub fn completions(&self) -> &Example {
        &self.example
    }
}

impl From<Example> for Prediction {
    fn from(example: Example) -> Self {
        Self::from_example(example)
    }
}
