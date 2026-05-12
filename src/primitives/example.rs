use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An untyped key-value container analogous to DSPy's `Example`.
///
/// Holds field values as `serde_json::Value` so it can represent any
/// signature's inputs and outputs without compile-time type coupling.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Example {
    fields: HashMap<String, Value>,
    #[serde(skip)]
    input_keys: Option<Vec<String>>,
}

impl Example {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_map(fields: HashMap<String, Value>) -> Self {
        Self {
            fields,
            input_keys: None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.fields.get(key)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.fields.get(key).and_then(|v| v.as_str())
    }

    pub fn set(&mut self, key: &str, value: Value) {
        self.fields.insert(key.to_string(), value);
    }

    pub fn with(mut self, key: &str, value: Value) -> Self {
        self.set(key, value);
        self
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.fields.keys()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.fields.iter()
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.fields.contains_key(key)
    }

    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.fields.remove(key)
    }

    pub fn with_inputs(mut self, keys: Vec<String>) -> Self {
        self.input_keys = Some(keys);
        self
    }

    pub fn input_keys(&self) -> Option<&[String]> {
        self.input_keys.as_deref()
    }

    pub fn inputs(&self) -> Example {
        match &self.input_keys {
            Some(keys) => {
                let fields: HashMap<_, _> = keys
                    .iter()
                    .filter_map(|k| self.fields.get(k).map(|v| (k.clone(), v.clone())))
                    .collect();
                Example {
                    fields,
                    input_keys: self.input_keys.clone(),
                }
            }
            None => self.clone(),
        }
    }

    pub fn to_map(&self) -> &HashMap<String, Value> {
        &self.fields
    }

    pub fn into_map(self) -> HashMap<String, Value> {
        self.fields
    }
}

impl From<HashMap<String, Value>> for Example {
    fn from(fields: HashMap<String, Value>) -> Self {
        Self::from_map(fields)
    }
}

impl FromIterator<(String, Value)> for Example {
    fn from_iter<I: IntoIterator<Item = (String, Value)>>(iter: I) -> Self {
        Self::from_map(iter.into_iter().collect())
    }
}
