mod common;

use common::{setup_mock, s, QA};
use dspy_rust::{
    Example, Module, Predict, ChainOfThought, Evaluate, Teleprompter,
    LabeledFewShot, exact_match, Settings,
};
use dspy_rust::primitives::Parameter;
use std::sync::Arc;
use serde_json;

#[tokio::test(flavor = "multi_thread")]
async fn e2e_predict_pipeline() {
    setup_mock(vec!["[[ ## answer ## ]]\nParis"]);

    let predict = Predict::<QA>::new();
    let input = Example::new().with("question", s("Capital of France?"));
    let result = predict.forward(&input).await.unwrap();

    assert_eq!(result.get_str("answer"), Some("Paris"));
}

#[tokio::test(flavor = "multi_thread")]
async fn e2e_chain_of_thought_pipeline() {
    setup_mock(vec![
        "[[ ## reasoning ## ]]\nLet me think step by step. France is in Europe. Paris is its capital.\n\n[[ ## answer ## ]]\nParis"
    ]);

    let cot = ChainOfThought::<QA>::new();
    let input = Example::new().with("question", s("Capital of France?"));
    let result = cot.forward(&input).await.unwrap();

    assert_eq!(result.get_str("answer"), Some("Paris"));
    assert!(result.get_str("reasoning").unwrap().contains("step by step"));
}

#[tokio::test(flavor = "multi_thread")]
async fn e2e_optimize_then_predict() {
    // Responses: 1 for the optimized forward call
    setup_mock(vec!["[[ ## answer ## ]]\nParis"]);

    let trainset = vec![
        Example::new().with("question", s("Capital of Italy?")).with("answer", s("Rome")),
        Example::new().with("question", s("Capital of Spain?")).with("answer", s("Madrid")),
    ];

    let mut predict = Predict::<QA>::new();
    let optimizer = LabeledFewShot::new(2);
    optimizer.compile(&mut predict, &trainset).await.unwrap();

    assert_eq!(predict.demos.len(), 2);

    let input = Example::new().with("question", s("Capital of France?"));
    let result = predict.forward(&input).await.unwrap();
    assert_eq!(result.get_str("answer"), Some("Paris"));
}

#[tokio::test(flavor = "multi_thread")]
async fn e2e_evaluate_dataset() {
    setup_mock(vec![
        "[[ ## answer ## ]]\nParis",
        "[[ ## answer ## ]]\nBerlin",
        "[[ ## answer ## ]]\nWRONG",
    ]);

    let dataset = vec![
        Example::new().with("question", s("Capital of France?")).with("answer", s("Paris")),
        Example::new().with("question", s("Capital of Germany?")).with("answer", s("Berlin")),
        Example::new().with("question", s("Capital of Spain?")).with("answer", s("Madrid")),
    ];

    let predict = Predict::<QA>::new();
    let metric = exact_match("answer");
    let evaluator = Evaluate::new().with_threads(1);
    let result = evaluator.run(&predict, &dataset, &metric).await;

    assert_eq!(result.total, 3);
    assert_eq!(result.errors, 0);
    // 2 out of 3 correct
    assert!((result.score - 2.0 / 3.0).abs() < 0.01);
}

#[tokio::test(flavor = "multi_thread")]
async fn e2e_state_save_load_roundtrip() {
    setup_mock(vec![
        "[[ ## answer ## ]]\nParis",
        "[[ ## answer ## ]]\nParis",
    ]);

    // Build and optimize a predict
    let mut predict = Predict::<QA>::new().with_instruction("Be concise.");
    predict.demos = vec![
        Example::new().with("question", s("Q1")).with("answer", s("A1")),
    ];

    // Save state
    let state = Parameter::dump_state(&predict);
    let state_json = serde_json::to_string_pretty(&state).unwrap();

    // Create a fresh predict and load state
    let mut predict2 = Predict::<QA>::new();
    let loaded_state: serde_json::Value = serde_json::from_str(&state_json).unwrap();
    Parameter::load_state(&mut predict2, &loaded_state).unwrap();

    assert_eq!(predict2.instruction.as_deref(), Some("Be concise."));
    assert_eq!(predict2.demos.len(), 1);

    // Both should produce the same result
    let input = Example::new().with("question", s("Capital of France?"));
    let r1 = predict.forward(&input).await.unwrap();
    let r2 = predict2.forward(&input).await.unwrap();
    assert_eq!(r1.get_str("answer"), r2.get_str("answer"));
}

#[tokio::test(flavor = "multi_thread")]
async fn e2e_scoped_settings_context() {
    let mock1 = Arc::new(common::MockLM::new(vec!["[[ ## answer ## ]]\nfrom_global"]));
    let mock2 = Arc::new(common::MockLM::new(vec!["[[ ## answer ## ]]\nfrom_scoped"]));

    dspy_rust::configure(Settings {
        lm: Some(mock1.clone()),
        ..Default::default()
    });

    let predict = Predict::<QA>::new();
    let input = Example::new().with("question", s("test"));

    // Inside scoped context, should use mock2
    let result = dspy_rust::context(
        Settings {
            lm: Some(mock2.clone()),
            ..Default::default()
        },
        predict.forward(&input),
    )
    .await
    .unwrap();

    assert_eq!(result.get_str("answer"), Some("from_scoped"));
    assert_eq!(mock2.call_count(), 1);
    assert_eq!(mock1.call_count(), 0);
}
