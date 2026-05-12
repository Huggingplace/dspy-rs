mod common;

use common::{setup_mock, s, QA, MockCodeExecutor};
use dspy_rust::{Example, Module, CodeAct, ProgramOfThought};

#[tokio::test(flavor = "multi_thread")]
async fn program_of_thought_single_shot() {
    setup_mock(vec![
        "```python\nprint(2 + 2)\n```",
    ]);

    let executor = MockCodeExecutor::new("4");
    let pot = ProgramOfThought::<QA>::new(executor);
    let input = Example::new().with("question", s("What is 2+2?"));
    let result = pot.forward(&input).await.unwrap();

    assert_eq!(result.get_str("answer"), Some("4"));
}

#[tokio::test(flavor = "multi_thread")]
async fn program_of_thought_no_code_block_errors() {
    setup_mock(vec![
        "I don't know how to code that.",
    ]);

    let executor = MockCodeExecutor::new("");
    let pot = ProgramOfThought::<QA>::new(executor);
    let input = Example::new().with("question", s("test"));
    let result = pot.forward(&input).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("No code block"));
}

#[tokio::test(flavor = "multi_thread")]
async fn code_act_multi_turn() {
    setup_mock(vec![
        // Turn 1: LM writes code
        "Let me compute this.\n```python\nprint(2+2)\n```",
        // Turn 2: LM sees result and gives final answer
        "[[ ## answer ## ]]\n4",
    ]);

    let executor = MockCodeExecutor::new("4");
    let code_act = CodeAct::<QA>::new(executor);
    let input = Example::new().with("question", s("What is 2+2?"));
    let result = code_act.forward(&input).await.unwrap();

    assert_eq!(result.get_str("answer"), Some("4"));
}

#[tokio::test(flavor = "multi_thread")]
async fn code_act_exceeds_max_iters() {
    setup_mock(vec![
        "```python\nprint('try 1')\n```",
        "```python\nprint('try 2')\n```",
    ]);

    let executor = MockCodeExecutor::new("not the answer");
    let code_act = CodeAct::<QA>::new(executor).with_max_iters(2);
    let input = Example::new().with("question", s("test"));
    let result = code_act.forward(&input).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("exceeded max iterations"));
}
