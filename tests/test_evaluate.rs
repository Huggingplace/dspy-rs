mod common;

use common::{setup_mock, s, QA};
use dspy_rust::{Example, Prediction, Predict, Evaluate, Metric, exact_match};
use dspy_rust::evaluate::FnMetric;

#[test]
fn exact_match_scores_correctly() {
    let metric = exact_match("answer");

    let ex = Example::new().with("answer", s("Paris"));
    let pred = Prediction::from_example(Example::new().with("answer", s("Paris")));
    assert_eq!(metric.score(&ex, &pred), 1.0);

    let pred_wrong = Prediction::from_example(Example::new().with("answer", s("London")));
    assert_eq!(metric.score(&ex, &pred_wrong), 0.0);
}

#[test]
fn exact_match_case_insensitive() {
    let metric = exact_match("answer");
    let ex = Example::new().with("answer", s("Paris"));
    let pred = Prediction::from_example(Example::new().with("answer", s("paris")));
    assert_eq!(metric.score(&ex, &pred), 1.0);
}

#[test]
fn exact_match_trims_whitespace() {
    let metric = exact_match("answer");
    let ex = Example::new().with("answer", s("Paris"));
    let pred = Prediction::from_example(Example::new().with("answer", s("  Paris  ")));
    assert_eq!(metric.score(&ex, &pred), 1.0);
}

#[test]
fn exact_match_missing_field_scores_zero() {
    let metric = exact_match("answer");
    let ex = Example::new().with("answer", s("Paris"));
    let pred = Prediction::from_example(Example::new());
    assert_eq!(metric.score(&ex, &pred), 0.0);
}

#[test]
fn fn_metric_works() {
    let metric = FnMetric(|_ex: &Example, pred: &Prediction| {
        if pred.get_str("answer") == Some("42") { 1.0 } else { 0.0 }
    });

    let ex = Example::new();
    let pred = Prediction::from_example(Example::new().with("answer", s("42")));
    assert_eq!(metric.score(&ex, &pred), 1.0);
}

#[tokio::test(flavor = "multi_thread")]
async fn evaluate_run_computes_score() {
    setup_mock(vec![
        "[[ ## answer ## ]]\nParis",
        "[[ ## answer ## ]]\nBerlin",
        "[[ ## answer ## ]]\nMadrid",
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
    assert!((result.score - 1.0).abs() < 0.01); // All correct
}

#[tokio::test(flavor = "multi_thread")]
async fn evaluate_handles_partial_correctness() {
    setup_mock(vec![
        "[[ ## answer ## ]]\nParis",
        "[[ ## answer ## ]]\nWRONG",
    ]);

    let dataset = vec![
        Example::new().with("question", s("Capital of France?")).with("answer", s("Paris")),
        Example::new().with("question", s("Capital of Germany?")).with("answer", s("Berlin")),
    ];

    let predict = Predict::<QA>::new();
    let metric = exact_match("answer");
    let evaluator = Evaluate::new().with_threads(1);
    let result = evaluator.run(&predict, &dataset, &metric).await;

    assert_eq!(result.total, 2);
    assert!((result.score - 0.5).abs() < 0.01); // 1 out of 2 correct
}
