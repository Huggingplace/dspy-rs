use crate::primitives::{Example, Prediction};

/// A metric scores a prediction against a ground-truth example.
pub trait Metric: Send + Sync {
    fn score(&self, example: &Example, prediction: &Prediction) -> f64;
}

/// Simple exact-match metric: compares a specific field for string equality.
pub struct ExactMatch {
    pub field: String,
}

impl ExactMatch {
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
        }
    }
}

impl Metric for ExactMatch {
    fn score(&self, example: &Example, prediction: &Prediction) -> f64 {
        let expected = example.get(&self.field).and_then(|v| v.as_str());
        let actual = prediction.get(&self.field).and_then(|v| v.as_str());
        match (expected, actual) {
            (Some(e), Some(a)) => {
                if e.trim().eq_ignore_ascii_case(a.trim()) {
                    1.0
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }
}

/// Convenience constructor for ExactMatch.
pub fn exact_match(field: &str) -> ExactMatch {
    ExactMatch::new(field)
}

/// Function-based metric wrapper.
pub struct FnMetric<F>(pub F);

impl<F> Metric for FnMetric<F>
where
    F: Fn(&Example, &Prediction) -> f64 + Send + Sync,
{
    fn score(&self, example: &Example, prediction: &Prediction) -> f64 {
        (self.0)(example, prediction)
    }
}
