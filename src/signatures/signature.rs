use crate::primitives::Example;
use crate::signatures::FieldDescriptor;

/// Trait implemented by `#[derive(Signature)]` on user-defined structs.
///
/// Provides introspection into the signature's input/output fields,
/// its instruction text, and the ability to generate a default instruction
/// from field names.
pub trait SignatureFields: serde::Serialize + serde::de::DeserializeOwned + Send + Sync {
    fn instruction() -> &'static str;
    fn input_fields() -> Vec<FieldDescriptor>;
    fn output_fields() -> Vec<FieldDescriptor>;
    fn signature_name() -> &'static str;

    fn default_instruction() -> String {
        let inputs: Vec<_> = Self::input_fields()
            .iter()
            .map(|f| format!("`{}`", f.name))
            .collect();
        let outputs: Vec<_> = Self::output_fields()
            .iter()
            .map(|f| format!("`{}`", f.name))
            .collect();
        format!(
            "Given the fields {}, produce the fields {}.",
            inputs.join(", "),
            outputs.join(", ")
        )
    }

    fn effective_instruction() -> String {
        let doc = Self::instruction();
        if doc.is_empty() {
            Self::default_instruction()
        } else {
            doc.to_string()
        }
    }
}

/// Convert between a Signature struct and an Example (untyped key-value map).
pub trait FromExample: Sized {
    fn from_example(example: &Example) -> Option<Self>;
    fn to_example(&self) -> Example;
}
