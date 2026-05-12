use regex::Regex;

use crate::clients::Message;
use crate::primitives::Example;
use crate::signatures::FieldDescriptor;

use super::Adapter;

/// Adapter that uses XML tags to delimit fields in the LM prompt and response.
///
/// Some models (especially Anthropic's Claude) perform well with XML-structured
/// prompts. This adapter wraps each field in `<field_name>...</field_name>` tags.
pub struct XmlAdapter;

impl XmlAdapter {
    fn format_field(name: &str, value: &str) -> String {
        format!("<{name}>\n{value}\n</{name}>")
    }

    fn format_demo(
        &self,
        input_fields: &[FieldDescriptor],
        output_fields: &[FieldDescriptor],
        demo: &Example,
    ) -> String {
        let mut parts = Vec::new();
        for field in input_fields.iter().chain(output_fields.iter()) {
            if let Some(val) = demo.get(field.name) {
                parts.push(Self::format_field(field.name, &val_to_string(val)));
            }
        }
        format!("<example>\n{}\n</example>", parts.join("\n"))
    }
}

#[async_trait::async_trait]
impl Adapter for XmlAdapter {
    fn format_messages(
        &self,
        instruction: &str,
        input_fields: &[FieldDescriptor],
        output_fields: &[FieldDescriptor],
        demos: &[Example],
        inputs: &Example,
    ) -> Vec<Message> {
        let mut system_parts = vec![instruction.to_string()];
        system_parts.push(String::new());
        system_parts.push("<format_description>".to_string());
        system_parts.push("Input fields:".to_string());
        for field in input_fields {
            system_parts.push(format!("  <{}> — {}", field.name, field.display_desc()));
        }
        system_parts.push("Output fields:".to_string());
        for field in output_fields {
            system_parts.push(format!("  <{}> — {}", field.name, field.display_desc()));
        }
        system_parts.push("</format_description>".to_string());

        let mut messages = vec![Message::system(system_parts.join("\n"))];

        if !demos.is_empty() {
            let demos_text = demos
                .iter()
                .map(|d| self.format_demo(input_fields, output_fields, d))
                .collect::<Vec<_>>()
                .join("\n\n");
            messages.push(Message::user(format!("<examples>\n{demos_text}\n</examples>")));
            messages.push(Message::assistant("I understand the format from the examples.".to_string()));
        }

        let mut input_parts = Vec::new();
        for field in input_fields {
            if let Some(val) = inputs.get(field.name) {
                input_parts.push(Self::format_field(field.name, &val_to_string(val)));
            }
        }
        let output_tags: Vec<_> = output_fields
            .iter()
            .map(|f| format!("<{}>", f.name))
            .collect();
        input_parts.push(format!(
            "Please respond with the following XML tags: {}",
            output_tags.join(", ")
        ));
        messages.push(Message::user(input_parts.join("\n")));

        messages
    }

    fn parse_response(
        &self,
        response: &str,
        output_fields: &[FieldDescriptor],
    ) -> anyhow::Result<Example> {
        let mut result = Example::new();

        for field in output_fields {
            let pattern = format!(r"<{}>\s*([\s\S]*?)\s*</{}>", field.name, field.name);
            let re = Regex::new(&pattern)
                .map_err(|e| anyhow::anyhow!("Invalid regex for field {}: {e}", field.name))?;

            if let Some(caps) = re.captures(response) {
                let value = caps[1].trim().to_string();
                result.set(field.name, serde_json::Value::String(value));
            } else if output_fields.len() == 1 {
                result.set(
                    field.name,
                    serde_json::Value::String(response.trim().to_string()),
                );
            } else {
                anyhow::bail!(
                    "Missing output field <{}> in XML response",
                    field.name
                );
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
