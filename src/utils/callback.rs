use async_trait::async_trait;
use serde_json::Value;

#[async_trait]
pub trait Callback: Send + Sync {
    async fn on_module_start(&self, module_name: &str, input: &Value) {
        let _ = (module_name, input);
    }

    async fn on_module_end(&self, module_name: &str, output: &Value) {
        let _ = (module_name, output);
    }

    async fn on_lm_request(&self, model: &str, messages: &Value) {
        let _ = (model, messages);
    }

    async fn on_lm_response(&self, model: &str, response: &Value) {
        let _ = (model, response);
    }
}
