# dspy-rust

A Rust port of [Stanford DSPy](https://github.com/stanfordnlp/dspy) -- programming (not prompting) foundation models.

## What is DSPy?

DSPy replaces hand-written prompts with declarative **signatures** and composable **modules**. An automatic **optimizer** (teleprompter) tunes prompts, demos, and instructions against a metric -- so you program LM behavior instead of engineering prompts.

## Quick Start

```rust
use dspy_rust::*;

#[derive(Signature)]
/// Given a question, produce a concise answer.
struct QA {
    #[input(desc = "the question to answer")]
    question: String,
    #[output(desc = "a concise factual answer")]
    answer: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let lm = OpenAILM::new("gpt-4o-mini");
    configure(Settings { lm: Some(std::sync::Arc::new(lm)), ..Default::default() });

    let predict = Predict::<QA>::new();
    let input = Example::new().with("question", "What is the capital of France?");
    let result = predict.forward(&input).await?;
    println!("{}", result.get_str("answer").unwrap_or_default());
    Ok(())
}
```

## Modules

| Module | Description |
|---|---|
| `Predict` | Basic signature-to-LM call |
| `ChainOfThought` | Injects a reasoning step before the answer |
| `ReAct` | Tool-use loop (Thought / Action / Observation) |
| `CodeAct` | Multi-turn code generation + execution |
| `ProgramOfThought` | Single-shot code generation + execution |
| `Parallel` | Run multiple modules concurrently |
| `Retry` | Retry with validator feedback |
| `Refine` | Iterative self-improvement |
| `BestOfN` | Sample N, return highest-scoring |
| `Streamify` | Streaming token output wrapper |

## Optimizers

| Optimizer | Strategy |
|---|---|
| `LabeledFewShot` | Select demos from labeled examples |
| `BootstrapFewShot` | Generate demos via teacher, filter by metric |
| `BootstrapFewShotWithRandomSearch` | Bootstrap N times, pick best on dev set |
| `MIPROv2` | Joint instruction + demo optimization |
| `COPRO` | Instruction-only optimization |
| `SIMBA` | Self-improving iterative bootstrapping |
| `BetterTogether` | Combined instruction + demo optimization |
| `BootstrapFinetune` | Export successful traces as fine-tuning data |
| `Ensemble` | Combine multiple optimized modules |

## Adapters

- `ChatAdapter` -- DSPy header format (`[[ ## field ## ]]`)
- `JsonAdapter` -- Structured JSON output
- `XmlAdapter` -- XML tag format (works well with Claude)

## Retrieval

- `EmbeddingRetriever` -- Cosine similarity over precomputed embeddings
- `KNN` -- Retriever wrapper for use in pipelines

## License

MIT
