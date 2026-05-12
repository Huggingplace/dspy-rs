use async_trait::async_trait;
use serde_json::Value;

use crate::primitives::Example;

use super::Retriever;

/// Trait for computing text embeddings.
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;
}

/// Embedding-based retriever: stores documents with precomputed embeddings,
/// retrieves the k most similar to a query via cosine similarity.
pub struct EmbeddingRetriever {
    embedder: Box<dyn Embedder>,
    documents: Vec<Example>,
    embeddings: Vec<Vec<f32>>,
    text_field: String,
}

impl EmbeddingRetriever {
    pub fn new(embedder: impl Embedder + 'static) -> Self {
        Self {
            embedder: Box::new(embedder),
            documents: Vec::new(),
            embeddings: Vec::new(),
            text_field: "text".to_string(),
        }
    }

    pub fn with_text_field(mut self, field: impl Into<String>) -> Self {
        self.text_field = field.into();
        self
    }

    pub async fn index(&mut self, documents: Vec<Example>) -> anyhow::Result<()> {
        let texts: Vec<String> = documents
            .iter()
            .map(|doc| {
                doc.get(&self.text_field)
                    .and_then(|v| match v {
                        Value::String(s) => Some(s.clone()),
                        _ => Some(v.to_string()),
                    })
                    .unwrap_or_default()
            })
            .collect();

        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.embeddings = self.embedder.embed(&text_refs).await?;
        self.documents = documents;
        Ok(())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[async_trait]
impl Retriever for EmbeddingRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> anyhow::Result<Vec<Example>> {
        if self.documents.is_empty() {
            return Ok(Vec::new());
        }

        let query_embedding = self.embedder.embed(&[query]).await?;
        let query_vec = query_embedding
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Embedder returned empty result"))?;

        let mut scored: Vec<(usize, f32)> = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, cosine_similarity(&query_vec, emb)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let results: Vec<Example> = scored
            .into_iter()
            .take(k)
            .map(|(idx, score)| {
                let mut doc = self.documents[idx].clone();
                doc.set("score", Value::Number(serde_json::Number::from_f64(score as f64).unwrap_or(serde_json::Number::from(0))));
                doc
            })
            .collect();

        Ok(results)
    }
}

/// KNN module: wraps a Retriever for use in a DSPy pipeline.
///
/// Takes a query field from the input Example, retrieves k passages,
/// and adds them to the output as a "passages" field.
pub struct KNN {
    retriever: Box<dyn Retriever>,
    query_field: String,
    k: usize,
}

impl KNN {
    pub fn new(retriever: impl Retriever + 'static) -> Self {
        Self {
            retriever: Box::new(retriever),
            query_field: "question".to_string(),
            k: 3,
        }
    }

    pub fn with_query_field(mut self, field: impl Into<String>) -> Self {
        self.query_field = field.into();
        self
    }

    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    pub async fn forward(&self, input: &Example) -> anyhow::Result<Vec<Example>> {
        let query = input
            .get(&self.query_field)
            .and_then(|v| match v {
                Value::String(s) => Some(s.clone()),
                _ => Some(v.to_string()),
            })
            .ok_or_else(|| {
                anyhow::anyhow!("Missing query field '{}' in input", self.query_field)
            })?;

        self.retriever.retrieve(&query, self.k).await
    }
}
