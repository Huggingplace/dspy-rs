mod common;

use dspy_rust::adapters::{Adapter, ChatAdapter, JsonAdapter, XmlAdapter};
use dspy_rust::signatures::FieldDescriptor;
use dspy_rust::{Example, Message};
use common::s;

fn qa_input_fields() -> Vec<FieldDescriptor> {
    vec![FieldDescriptor {
        name: "question",
        desc: "the question",
        prefix: "",
        type_name: "String",
    }]
}

fn qa_output_fields() -> Vec<FieldDescriptor> {
    vec![FieldDescriptor {
        name: "answer",
        desc: "the answer",
        prefix: "",
        type_name: "String",
    }]
}

fn multi_output_fields() -> Vec<FieldDescriptor> {
    vec![
        FieldDescriptor { name: "label", desc: "category", prefix: "", type_name: "String" },
        FieldDescriptor { name: "confidence", desc: "score", prefix: "", type_name: "String" },
    ]
}

// ─── ChatAdapter ───

#[test]
fn chat_adapter_format_messages_has_headers() {
    let adapter = ChatAdapter;
    let input = Example::new().with("question", s("What is 2+2?"));
    let msgs = adapter.format_messages(
        "Answer questions.",
        &qa_input_fields(),
        &qa_output_fields(),
        &[],
        &input,
    );

    assert!(msgs.len() >= 2);
    let system = &msgs[0].content;
    assert!(system.contains("[[ ## question ## ]]"));
    assert!(system.contains("[[ ## answer ## ]]"));
}

#[test]
fn chat_adapter_format_includes_demos() {
    let adapter = ChatAdapter;
    let demo = Example::new().with("question", s("Q1")).with("answer", s("A1"));
    let input = Example::new().with("question", s("Q2"));
    let msgs = adapter.format_messages(
        "Answer.",
        &qa_input_fields(),
        &qa_output_fields(),
        &[demo],
        &input,
    );

    // system + demo_user + demo_assistant + user = 4 messages
    assert!(msgs.len() >= 4);
}

#[test]
fn chat_adapter_parse_response_with_headers() {
    let adapter = ChatAdapter;
    let response = "[[ ## answer ## ]]\nParis";
    let result = adapter.parse_response(response, &qa_output_fields()).unwrap();
    assert_eq!(result.get_str("answer"), Some("Paris"));
}

#[test]
fn chat_adapter_parse_single_field_fallback() {
    let adapter = ChatAdapter;
    let response = "Paris";
    let result = adapter.parse_response(response, &qa_output_fields()).unwrap();
    assert_eq!(result.get_str("answer"), Some("Paris"));
}

#[test]
fn chat_adapter_parse_multi_field() {
    let adapter = ChatAdapter;
    let response = "[[ ## label ## ]]\npositive\n\n[[ ## confidence ## ]]\n0.95";
    let result = adapter.parse_response(response, &multi_output_fields()).unwrap();
    assert_eq!(result.get_str("label"), Some("positive"));
    assert_eq!(result.get_str("confidence"), Some("0.95"));
}

#[test]
fn chat_adapter_parse_multi_field_missing_errors() {
    let adapter = ChatAdapter;
    let response = "[[ ## label ## ]]\npositive";
    let result = adapter.parse_response(response, &multi_output_fields());
    assert!(result.is_err());
}

// ─── JsonAdapter ───

#[test]
fn json_adapter_parse_valid_json() {
    let adapter = JsonAdapter;
    let response = r#"{"answer": "Paris"}"#;
    let result = adapter.parse_response(response, &qa_output_fields()).unwrap();
    assert_eq!(result.get_str("answer"), Some("Paris"));
}

#[test]
fn json_adapter_parse_json_in_code_fence() {
    let adapter = JsonAdapter;
    let response = "```json\n{\"answer\": \"Paris\"}\n```";
    let result = adapter.parse_response(response, &qa_output_fields()).unwrap();
    assert_eq!(result.get_str("answer"), Some("Paris"));
}

#[test]
fn json_adapter_parse_multi_field() {
    let adapter = JsonAdapter;
    let response = r#"{"label": "positive", "confidence": "0.95"}"#;
    let result = adapter.parse_response(response, &multi_output_fields()).unwrap();
    assert_eq!(result.get_str("label"), Some("positive"));
    assert_eq!(result.get_str("confidence"), Some("0.95"));
}

// ─── XmlAdapter ───

#[test]
fn xml_adapter_parse_response() {
    let adapter = XmlAdapter;
    let response = "<answer>\nParis\n</answer>";
    let result = adapter.parse_response(response, &qa_output_fields()).unwrap();
    assert_eq!(result.get_str("answer"), Some("Paris"));
}

#[test]
fn xml_adapter_parse_multi_field() {
    let adapter = XmlAdapter;
    let response = "<label>\npositive\n</label>\n<confidence>\n0.95\n</confidence>";
    let result = adapter.parse_response(response, &multi_output_fields()).unwrap();
    assert_eq!(result.get_str("label"), Some("positive"));
    assert_eq!(result.get_str("confidence"), Some("0.95"));
}

#[test]
fn xml_adapter_single_field_fallback() {
    let adapter = XmlAdapter;
    let response = "Paris";
    let result = adapter.parse_response(response, &qa_output_fields()).unwrap();
    assert_eq!(result.get_str("answer"), Some("Paris"));
}

#[test]
fn xml_adapter_multi_field_missing_errors() {
    let adapter = XmlAdapter;
    let response = "<label>positive</label>";
    let result = adapter.parse_response(response, &multi_output_fields());
    assert!(result.is_err());
}

#[test]
fn xml_adapter_format_messages() {
    let adapter = XmlAdapter;
    let input = Example::new().with("question", s("What is 2+2?"));
    let msgs = adapter.format_messages(
        "Answer questions.",
        &qa_input_fields(),
        &qa_output_fields(),
        &[],
        &input,
    );

    assert!(msgs.len() >= 2);
    let system = &msgs[0].content;
    assert!(system.contains("<question>"));
    assert!(system.contains("<answer>"));
}
