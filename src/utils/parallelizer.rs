use std::future::Future;

use futures::stream::{self, StreamExt};

/// Execute a batch of async tasks with bounded concurrency.
pub async fn run_parallel<T, F, Fut>(items: Vec<T>, concurrency: usize, f: F) -> Vec<Fut::Output>
where
    T: Send + 'static,
    F: Fn(T) -> Fut + Send + Sync + 'static,
    Fut: Future + Send,
    Fut::Output: Send,
{
    stream::iter(items)
        .map(|item| f(item))
        .buffer_unordered(concurrency)
        .collect()
        .await
}
