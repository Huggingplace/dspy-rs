/// Metadata about a single input or output field in a Signature.
#[derive(Debug, Clone)]
pub struct FieldDescriptor {
    pub name: &'static str,
    pub desc: &'static str,
    pub prefix: &'static str,
    pub type_name: &'static str,
}

impl FieldDescriptor {
    pub fn display_desc(&self) -> &str {
        if self.desc.is_empty() {
            self.name
        } else {
            self.desc
        }
    }

    pub fn header(&self) -> String {
        format!("[[ ## {} ## ]]", self.name)
    }
}

/// Marker type for input fields (used in builder APIs, not stored at runtime).
pub struct InputField;

/// Marker type for output fields.
pub struct OutputField;
