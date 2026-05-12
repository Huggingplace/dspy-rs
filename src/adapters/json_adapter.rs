use crate::clients::Message;
use crate::primitives::Example;
use crate::signatures::FieldDescriptor;

use super::Adapter;

/// Adapter that requests structured JSON output from the LM.
pub struct JsonAdapter;

#[async_trait::async_trait]
impl Adapter for JsonAdapter {
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

        let mut schema_fields = Vec::new();
        for field in output_fields {
            schema_fields.push(format!(
                "  \"{}\": \"<{}>\"",
                field.name,
                field.display_desc()
            ));
        }
        system_parts.push("Respond with a JSON object with the following fields:".to_string());
        system_parts.push(format!("{{\n{}\n}}", schema_fields.join(",\n")));

        let mut messages = vec![Message::system(system_parts.join("\n"))];

        for demo in demos {
            let mut input_parts = Vec::new();
            for field in input_fields {
                if let Some(val) = demo.get(field.name) {
                    input_parts.push(format!("{}: {}", field.name, val_to_string(val)));
                }
            }
            messages.push(Message::user(input_parts.join("\n")));

            let mut output_obj = serde_json::Map::new();
            for field in output_fields {
                if let Some(val) = demo.get(field.name) {
                    output_obj.insert(field.name.to_string(), val.clone());
                }
            }
            messages.push(Message::assistant(
                serde_json::to_string(&serde_json::Value::Object(output_obj))
                    .unwrap_or_default(),
            ));
        }

        let mut input_parts = Vec::new();
        for field in input_fields {
            if let Some(val) = inputs.get(field.name) {
                input_parts.push(format!("{}: {}", field.name, val_to_string(val)));
            }
        }
        messages.push(Message::user(input_parts.join("\n")));

        messages
    }

    fn parse_response(
        &self,
        response: &str,
        output_fields: &[FieldDescriptor],
    ) -> anyhow::Result<Example> {
        let trimmed = response.trim();
        let json_str = if trimmed.starts_with("```") {
            trimmed
                .trim_start_matches("```json")
                .trim_start_matches("```")
                .trim_end_matches("```")
                .trim()
        } else {
            trimmed
        };

        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| anyhow::anyhow!("Failed to parse JSON response: {e}\nRaw: {json_str}"))?;

        let obj = parsed
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Expected JSON object in response"))?;

        let mut result = Example::new();
        for field in output_fields {
            if let Some(val) = obj.get(field.name) {
                result.set(field.name, val.clone());
            } else {
                anyhow::bail!("Missing output field `{}` in JSON response", field.name);
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
