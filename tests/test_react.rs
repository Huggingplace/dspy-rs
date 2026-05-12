mod common;

use common::{setup_mock, s, QA};
use dspy_rust::{Example, Module, ReAct, Tool};

#[tokio::test(flavor = "multi_thread")]
async fn react_tool_loop_and_finish() {
    setup_mock(vec![
        // Step 1: LM asks to use a tool
        "Thought: I need to look up the capital.\nAction: lookup\nAction Input: France",
        // Step 2: LM finishes with the answer
        "Thought: I have enough information.\nAction: Finish\nAction Input: done\n\n[[ ## answer ## ]]\nParis",
    ]);

    let lookup = Tool::new("lookup", "Look up facts", |input: &str| {
        if input.contains("France") {
            "The capital of France is Paris.".to_string()
        } else {
            "Unknown".to_string()
        }
    });

    let react = ReAct::<QA>::new(vec![lookup]);
    let input = Example::new().with("question", s("What is the capital of France?"));
    let result = react.forward(&input).await.unwrap();

    assert_eq!(result.get_str("answer"), Some("Paris"));
}

#[tokio::test(flavor = "multi_thread")]
async fn react_unknown_tool_reports_error() {
    setup_mock(vec![
        // LM tries a tool that doesn't exist
        "Thought: Let me search.\nAction: google\nAction Input: France capital",
        // LM corrects and finishes
        "Thought: I have enough information.\nAction: Finish\nAction Input: done\n\n[[ ## answer ## ]]\nParis",
    ]);

    let lookup = Tool::new("lookup", "Look up facts", |_: &str| "result".to_string());

    let react = ReAct::<QA>::new(vec![lookup]);
    let input = Example::new().with("question", s("Capital of France?"));
    let result = react.forward(&input).await.unwrap();

    assert_eq!(result.get_str("answer"), Some("Paris"));
}

#[tokio::test(flavor = "multi_thread")]
async fn react_exceeds_max_iters() {
    setup_mock(vec![
        "Thought: thinking\nAction: lookup\nAction Input: x",
        "Thought: still thinking\nAction: lookup\nAction Input: y",
    ]);

    let tool = Tool::new("lookup", "look up", |_: &str| "result".to_string());
    let react = ReAct::<QA>::new(vec![tool]).with_max_iters(2);
    let input = Example::new().with("question", s("test"));
    let result = react.forward(&input).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("exceeded max iterations"));
}
