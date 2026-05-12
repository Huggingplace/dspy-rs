mod evaluate;
mod metrics;

pub use evaluate::{Evaluate, EvalResult};
pub use metrics::{Metric, ExactMatch, FnMetric, exact_match};
