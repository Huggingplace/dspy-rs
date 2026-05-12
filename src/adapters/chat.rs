use regex::Regex;

use crate::clients::Message;
use crate::primitives::Example;
use crate::signatures::FieldDescriptor;

use super::Adapter;

/// Default adapter using DSPy's `[[ ## field_name ## ]]` header format.
pub struct ChatAdapter;

impl ChatAdapter {
    fn format_field_header(name: &str) -> String {
        format!("[[ ## {} ## ]]", name)
    }

    fn format_field_with_value(field: &FieldDescriptor, value: &str) -> String {
        format!("{}\n{}", Self::format_field_header(field.name), value)
    }

    fn format_demo(
        &self,
        input_fields: &[FieldDescriptor],
        output_fields: &[FieldDescriptor],
        demo: &Example,
    ) -> String {
        let mut parts = Vec::new();
        for field in input_fields {
            if let Some(val) = demo.get(field.name) {
                let text = val_to_string(val);
                parts.push(Self::format_field_with_value(field, &text));
            }
        }
        for field in output_fields {
            if let Some(val) = demo.get(field.name) {
                let text = val_to_string(val);
                parts.push(Self::format_field_with_value(field, &text));
            }
        }
        parts.join("\n\n")
    }

    fn format_input(
        &self,
        input_fields: &[FieldDescriptor],
        output_fields: &[FieldDescriptor],
        inputs: &Example,
    ) -> String {
        let mut parts = Vec::new();
        for field in input_fields {
            if let Some(val) = inputs.get(field.name) {
                let text = val_to_string(val);
                parts.push(Self::format_field_with_value(field, &text));
            }
        }
        for field in output_fields {
            parts.push(format!(
                "{}\n",
                Self::format_field_header(field.name)
            ));
        }
        parts.join("\n\n")
    }
}

#[async_trait::async_trait]
impl Adapter for ChatAdapter {
    fn format_messages(
        &self,
        instruction: &str,
        input_fields: &[FieldDescriptor],
        output_fields: &[FieldDescriptor],
        demos: &[Example],
        inputs: &Example,
    ) -> Vec<Message> {
        let mut messages = Vec::new();

        let mut system_parts = vec![instruction.to_string()];
        system_parts.push(String::new());
        system_parts.push("---".to_string());
        system_parts.push(String::new());
        system_parts.push("Follow the following format.".to_string());
        system_parts.push(String::new());

        for field in input_fields {
            system_parts.push(format!(
                "{}: {}",
                Self::format_field_header(field.name),
                field.display_desc()
            ));
        }
        for field in output_fields {
            system_parts.push(format!(
                "{}: {}",
                Self::format_field_header(field.name),
                field.display_desc()
            ));
        }

        messages.push(Message::system(system_parts.join("\n")));

        for demo in demos {
            let demo_text = self.format_demo(input_fields, output_fields, demo);
            messages.push(Message::user(demo_text.clone()));
            messages.push(Message::assistant("(demo)".to_string()));
        }

        let user_text = self.format_input(input_fields, output_fields, inputs);
        messages.push(Message::user(user_text));

        messages
    }

    fn parse_response(
        &self,
        response: &str,
        output_fields: &[FieldDescriptor],
    ) -> anyhow::Result<Example> {
        let mut result = Example::new();
        let header_pattern = Regex::new(r"\[\[ ## (\w+) ## \]\]").unwrap();

        let mut current_field: Option<String> = None;
        let mut current_value = String::new();

        for line in response.lines() {
            if let Some(caps) = header_pattern.captures(line) {
                if let Some(field_name) = current_field.take() {
                    let trimmed = current_value.trim().to_string();
                    result.set(
                        &field_name,
                        serde_json::Value::String(trimmed),
                    );
                }
                current_field = Some(caps[1].to_string());
                current_value.clear();
            } else if current_field.is_some() {
                if !current_value.is_empty() {
                    current_value.push('\n');
                }
                current_value.push_str(line);
            }
        }

        if let Some(field_name) = current_field {
            let trimmed = current_value.trim().to_string();
            result.set(
                &field_name,
                serde_json::Value::String(trimmed),
            );
        }

        for field in output_fields {
            if !result.contains_key(field.name) {
                if output_fields.len() == 1 {
                    result.set(
                        field.name,
                        serde_json::Value::String(response.trim().to_string()),
                    );
                } else {
                    anyhow::bail!(
                        "Missing output field `{}` in LM response",
                        field.name
                    );
                }
            }
        }

        Ok(result)
    }
}

fn val_to_string(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}
