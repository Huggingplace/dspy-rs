use async_trait::async_trait;

use crate::primitives::{Example, Module};

/// Base trait for all DSPy optimizers.
///
/// A Teleprompter takes a Module and a training set, then produces an
/// optimized version of that Module (with better demos, instructions, etc.).
#[async_trait]
pub trait Teleprompter: Send + Sync {
    /// Compile (optimize) a module using the given training set.
    async fn compile(
        &self,
        module: &mut dyn Module,
        trainset: &[Example],
    ) -> anyhow::Result<()>;
}
