pub mod signatures;
pub mod primitives;
pub mod clients;
pub mod adapters;
pub mod predict;
pub mod evaluate;
pub mod teleprompt;
pub mod retrievers;
pub mod streaming;
pub mod utils;

pub use dspy_rust_macros::Signature;

// Core types
pub use signatures::{FieldDescriptor, FromExample, SignatureFields};
pub use primitives::{Example, Module, Parameter, Prediction};
pub use clients::{LM, LMConfig, LMResponse, Message, MessageRole, OpenAILM};
pub use utils::settings::{configure, context, Settings};

// Predict modules
pub use predict::{Predict, ChainOfThought, ReAct, Tool, Parallel, Retry, Refine, BestOfN, CodeAct, CodeExecutor, ProgramOfThought};

// Adapters
pub use adapters::{Adapter, ChatAdapter, JsonAdapter, XmlAdapter};

// Evaluation
pub use evaluate::{Evaluate, Metric, exact_match};

// Optimizers
pub use teleprompt::{
    Teleprompter, LabeledFewShot, BootstrapFewShot,
    BootstrapFewShotWithRandomSearch, MIPROv2, COPRO, Ensemble,
    SIMBA, BetterTogether, BootstrapFinetune,
};

// Retrievers
pub use retrievers::{Retriever, Embedder, EmbeddingRetriever, KNN};

// Streaming
pub use streaming::Streamify;
