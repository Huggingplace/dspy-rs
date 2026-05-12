mod common;

use common::{setup_mock, s, QA, Classify};
use dspy_rust::{Example, Module, Predict, ChainOfThought};

#[tokio::test(flavor = "multi_thread")]
async fn predict_forward_returns_answer() {
    let mock = setup_mock(vec!["[[ ## answer ## ]]\nParis"]);

    let predict = Predict::<QA>::new();
    let input = Example::new().with("question", s("What is the capital of France?"));
    let result = predict.forward(&input).await.unwrap();

    assert_eq!(result.get_str("answer"), Some("Paris"));
    assert_eq!(mock.call_count(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn predict_forward_single_field_fallback() {
    setup_mock(vec!["Paris"]);

    let predict = Predict::<QA>::new();
    let input = Example::new().with("question", s("Capital of France?"));
    let result = predict.forward(&input).await.unwrap();

    assert_eq!(result.get_str("answer"), Some("Paris"));
}

#[tokio::test(flavor = "multi_thread")]
async fn predict_messages_contain_question() {
    let mock = setup_mock(vec!["[[ ## answer ## ]]\nParis"]);

    let predict = Predict::<QA>::new();
    let input = Example::new().with("question", s("What is the capital of France?"));
    let _ = predict.forward(&input).await.unwrap();

    let msgs = mock.last_messages();
    let user_msg = msgs.iter().find(|m| m.content.contains("France")).unwrap();
    assert!(user_msg.content.contains("France"));
}

#[tokio::test(flavor = "multi_thread")]
async fn chain_of_thought_includes_reasoning() {
    setup_mock(vec![
        "[[ ## reasoning ## ]]\nFrance is a country in Europe. Its capital is Paris.\n\n[[ ## answer ## ]]\nParis"
    ]);

    let cot = ChainOfThought::<QA>::new();
    let input = Example::new().with("question", s("Capital of France?"));
    let result = cot.forward(&input).await.unwrap();

    assert_eq!(result.get_str("answer"), Some("Paris"));
    assert!(result.get_str("reasoning").is_some());
    assert!(result.get_str("reasoning").unwrap().contains("Europe"));
}

#[tokio::test(flavor = "multi_thread")]
async fn predict_with_custom_instruction() {
    let mock = setup_mock(vec!["[[ ## answer ## ]]\n42"]);

    let predict = Predict::<QA>::new()
        .with_instruction("Always answer with 42.");
    let input = Example::new().with("question", s("What?"));
    let result = predict.forward(&input).await.unwrap();

    assert_eq!(result.get_str("answer"), Some("42"));

    let msgs = mock.last_messages();
    let system = &msgs[0].content;
    assert!(system.contains("Always answer with 42"));
}

#[tokio::test(flavor = "multi_thread")]
async fn predict_with_demos() {
    let mock = setup_mock(vec!["[[ ## answer ## ]]\nBerlin"]);

    let demo = Example::new().with("question", s("Capital of Italy?")).with("answer", s("Rome"));
    let predict = Predict::<QA>::new().with_demos(vec![demo]);
    let input = Example::new().with("question", s("Capital of Germany?"));
    let result = predict.forward(&input).await.unwrap();

    assert_eq!(result.get_str("answer"), Some("Berlin"));

    let msgs = mock.last_messages();
    // With demos, there should be more than 2 messages (system + demo_user + demo_assistant + user)
    assert!(msgs.len() >= 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn predict_no_lm_configured_errors() {
    // Configure with no LM
    dspy_rust::configure(dspy_rust::Settings::default());

    let predict = Predict::<QA>::new();
    let input = Example::new().with("question", s("test"));
    let result = predict.forward(&input).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("No LM configured"));
}
