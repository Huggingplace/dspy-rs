mod embeddings;

use async_trait::async_trait;

use crate::primitives::Example;

/// A retriever fetches relevant passages/documents for a query.
#[async_trait]
pub trait Retriever: Send + Sync {
    async fn retrieve(&self, query: &str, k: usize) -> anyhow::Result<Vec<Example>>;
}

pub use embeddings::{Embedder, EmbeddingRetriever, KNN};
