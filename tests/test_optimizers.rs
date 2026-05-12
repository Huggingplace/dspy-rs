mod common;

use common::{setup_mock, s, QA};
use dspy_rust::{Example, Module, Predict, Teleprompter, LabeledFewShot, BootstrapFewShot, exact_match};
use dspy_rust::primitives::Parameter;

#[tokio::test(flavor = "multi_thread")]
async fn labeled_fewshot_assigns_demos() {
    setup_mock(vec![]);

    let trainset = vec![
        Example::new().with("question", s("Q1")).with("answer", s("A1")),
        Example::new().with("question", s("Q2")).with("answer", s("A2")),
        Example::new().with("question", s("Q3")).with("answer", s("A3")),
    ];

    let mut predict = Predict::<QA>::new();
    let optimizer = LabeledFewShot::new(2);
    optimizer.compile(&mut predict, &trainset).await.unwrap();

    assert_eq!(predict.demos.len(), 2);
    assert_eq!(predict.demos[0].get_str("question"), Some("Q1"));
    assert_eq!(predict.demos[1].get_str("question"), Some("Q2"));
}

#[tokio::test(flavor = "multi_thread")]
async fn labeled_fewshot_k_greater_than_trainset() {
    setup_mock(vec![]);

    let trainset = vec![
        Example::new().with("question", s("Q1")).with("answer", s("A1")),
    ];

    let mut predict = Predict::<QA>::new();
    let optimizer = LabeledFewShot::new(10);
    optimizer.compile(&mut predict, &trainset).await.unwrap();

    assert_eq!(predict.demos.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn bootstrap_fewshot_collects_demos() {
    // MockLM needs to respond to:
    // 1-3: forward() calls during bootstrap (one per training example)
    setup_mock(vec![
        "[[ ## answer ## ]]\nA1",
        "[[ ## answer ## ]]\nA2",
        "[[ ## answer ## ]]\nA3",
    ]);

    let trainset = vec![
        Example::new().with("question", s("Q1")).with("answer", s("A1")),
        Example::new().with("question", s("Q2")).with("answer", s("A2")),
        Example::new().with("question", s("Q3")).with("answer", s("A3")),
    ];

    let mut predict = Predict::<QA>::new();
    let optimizer = BootstrapFewShot::new(exact_match("answer"))
        .with_max_bootstrapped_demos(2)
        .with_max_labeled_demos(1);

    optimizer.compile(&mut predict, &trainset).await.unwrap();

    // Should have bootstrapped demos (up to 2) + labeled demos (up to 1)
    assert!(!predict.demos.is_empty());
    assert!(predict.demos.len() <= 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn predict_state_roundtrip_after_optimization() {
    setup_mock(vec![]);

    let trainset = vec![
        Example::new().with("question", s("Q1")).with("answer", s("A1")),
        Example::new().with("question", s("Q2")).with("answer", s("A2")),
    ];

    let mut predict = Predict::<QA>::new();
    let optimizer = LabeledFewShot::new(2);
    optimizer.compile(&mut predict, &trainset).await.unwrap();

    let state = Parameter::dump_state(&predict);
    let mut predict2 = Predict::<QA>::new();
    Parameter::load_state(&mut predict2, &state).unwrap();

    assert_eq!(predict2.demos.len(), predict.demos.len());
    assert_eq!(
        predict2.demos[0].get_str("question"),
        predict.demos[0].get_str("question")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn predict_reset_clears_state() {
    setup_mock(vec![]);

    let mut predict = Predict::<QA>::new()
        .with_demos(vec![Example::new().with("question", s("Q1"))])
        .with_instruction("custom");

    assert_eq!(predict.demos.len(), 1);
    assert!(predict.instruction.is_some());

    predict.reset();
    assert!(predict.demos.is_empty());
    assert!(predict.instruction.is_none());
}
