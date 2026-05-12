mod chat;
mod json_adapter;
mod xml_adapter;

pub use chat::ChatAdapter;
pub use json_adapter::JsonAdapter;
pub use xml_adapter::XmlAdapter;

use async_trait::async_trait;

use crate::clients::{LM, LMConfig, LMResponse, Message};
use crate::primitives::Example;
use crate::signatures::FieldDescriptor;

/// An Adapter formats a Signature's fields + demos into LM messages,
/// then parses the LM response back into field values.
#[async_trait]
pub trait Adapter: Send + Sync {
    fn format_messages(
        &self,
        instruction: &str,
        input_fields: &[FieldDescriptor],
        output_fields: &[FieldDescriptor],
        demos: &[Example],
        inputs: &Example,
    ) -> Vec<Message>;

    fn parse_response(
        &self,
        response: &str,
        output_fields: &[FieldDescriptor],
    ) -> anyhow::Result<Example>;

    async fn call(
        &self,
        lm: &dyn LM,
        lm_config: &LMConfig,
        instruction: &str,
        input_fields: &[FieldDescriptor],
        output_fields: &[FieldDescriptor],
        demos: &[Example],
        inputs: &Example,
    ) -> anyhow::Result<(Example, LMResponse)> {
        let messages = self.format_messages(
            instruction,
            input_fields,
            output_fields,
            demos,
            inputs,
        );
        let response = lm.complete(&messages, lm_config).await?;
        let parsed = self.parse_response(&response.content, output_fields)?;
        Ok((parsed, response))
    }
}
