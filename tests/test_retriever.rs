mod common;

use common::{s, MockEmbedder};
use dspy_rust::{Example, EmbeddingRetriever, KNN, Retriever};

#[tokio::test]
async fn embedding_retriever_index_and_retrieve() {
    let mut retriever = EmbeddingRetriever::new(MockEmbedder);

    let docs = vec![
        Example::new().with("text", s("short")),
        Example::new().with("text", s("a medium length text")),
        Example::new().with("text", s("this is a much longer document with many words")),
    ];

    retriever.index(docs).await.unwrap();

    // Query with similar length to "short" (5 chars)
    let results = retriever.retrieve("hello", 2).await.unwrap();
    assert_eq!(results.len(), 2);

    // Results should have a score field
    assert!(results[0].get("score").is_some());
}

#[tokio::test]
async fn embedding_retriever_empty_index() {
    let retriever = EmbeddingRetriever::new(MockEmbedder);
    let results = retriever.retrieve("anything", 5).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn embedding_retriever_k_larger_than_docs() {
    let mut retriever = EmbeddingRetriever::new(MockEmbedder);
    let docs = vec![Example::new().with("text", s("only one"))];
    retriever.index(docs).await.unwrap();

    let results = retriever.retrieve("query", 10).await.unwrap();
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn knn_forward_extracts_query() {
    let mut retriever = EmbeddingRetriever::new(MockEmbedder);
    let docs = vec![
        Example::new().with("text", s("Paris is the capital of France")),
        Example::new().with("text", s("Berlin is the capital of Germany")),
    ];
    retriever.index(docs).await.unwrap();

    let knn = KNN::new(retriever).with_query_field("question").with_k(1);
    let input = Example::new().with("question", s("What is the capital?"));
    let results = knn.forward(&input).await.unwrap();

    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn knn_missing_query_field_errors() {
    let retriever = EmbeddingRetriever::new(MockEmbedder);
    let knn = KNN::new(retriever).with_query_field("question");
    let input = Example::new().with("text", s("no question field"));
    let result = knn.forward(&input).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Missing query field"));
}

#[tokio::test]
async fn embedding_retriever_custom_text_field() {
    let mut retriever = EmbeddingRetriever::new(MockEmbedder)
        .with_text_field("content");

    let docs = vec![
        Example::new().with("content", s("Hello world")),
        Example::new().with("content", s("Goodbye world")),
    ];
    retriever.index(docs).await.unwrap();

    let results = retriever.retrieve("Hello", 1).await.unwrap();
    assert_eq!(results.len(), 1);
}
