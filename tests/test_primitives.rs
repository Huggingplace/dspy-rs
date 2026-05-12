mod common;

use dspy_rust::{Example, Prediction, Predict};
use dspy_rust::primitives::Parameter;
use dspy_rust::Module;
use common::s;

#[test]
fn example_set_get() {
    let mut ex = Example::new();
    ex.set("name", s("Alice"));
    assert_eq!(ex.get_str("name"), Some("Alice"));
    assert!(ex.get("missing").is_none());
}

#[test]
fn example_with_builder() {
    let ex = Example::new()
        .with("a", s("1"))
        .with("b", s("2"));
    assert_eq!(ex.len(), 2);
    assert_eq!(ex.get_str("a"), Some("1"));
    assert_eq!(ex.get_str("b"), Some("2"));
}

#[test]
fn example_inputs() {
    let ex = Example::new()
        .with("question", s("What?"))
        .with("answer", s("42"))
        .with_inputs(vec!["question".to_string()]);

    let inputs = ex.inputs();
    assert!(inputs.contains_key("question"));
    assert!(!inputs.contains_key("answer"));
}

#[test]
fn example_clone_independent() {
    let ex = Example::new().with("x", s("1"));
    let mut cloned = ex.clone();
    cloned.set("x", s("2"));
    assert_eq!(ex.get_str("x"), Some("1"));
    assert_eq!(cloned.get_str("x"), Some("2"));
}

#[test]
fn example_keys_and_iter() {
    let ex = Example::new().with("a", s("1")).with("b", s("2"));
    let keys: Vec<&String> = ex.keys().collect();
    assert_eq!(keys.len(), 2);
    let pairs: Vec<_> = ex.iter().collect();
    assert_eq!(pairs.len(), 2);
}

#[test]
fn example_remove() {
    let mut ex = Example::new().with("a", s("1")).with("b", s("2"));
    ex.remove("a");
    assert!(!ex.contains_key("a"));
    assert_eq!(ex.len(), 1);
}

#[test]
fn prediction_from_example() {
    let ex = Example::new().with("answer", s("Paris"));
    let pred = Prediction::from_example(ex);
    assert_eq!(pred.get_str("answer"), Some("Paris"));
    assert!(pred.completions().contains_key("answer"));
}

#[test]
fn prediction_set_and_get() {
    let mut pred = Prediction::new();
    pred.set("field", s("value"));
    assert_eq!(pred.get_str("field"), Some("value"));
}

#[tokio::test(flavor = "multi_thread")]
async fn predict_dump_load_state() {
    let mut predict = Predict::<common::QA>::new();
    predict.demos = vec![
        Example::new().with("question", s("Q1")).with("answer", s("A1")),
    ];
    predict.instruction = Some("Custom instruction".to_string());

    let state = Parameter::dump_state(&predict);
    assert_eq!(state.get("instruction").and_then(|v| v.as_str()), Some("Custom instruction"));

    let demos = state.get("demos").and_then(|v| v.as_array());
    assert_eq!(demos.map(|d| d.len()), Some(1));

    let mut predict2 = Predict::<common::QA>::new();
    Parameter::load_state(&mut predict2, &state).unwrap();
    assert_eq!(predict2.instruction.as_deref(), Some("Custom instruction"));
    assert_eq!(predict2.demos.len(), 1);
}

#[test]
fn example_serialize_deserialize() {
    let ex = Example::new().with("question", s("What?")).with("answer", s("42"));
    let json = serde_json::to_string(&ex).unwrap();
    let deserialized: Example = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.get_str("question"), Some("What?"));
    assert_eq!(deserialized.get_str("answer"), Some("42"));
}
