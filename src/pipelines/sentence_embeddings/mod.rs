pub mod builder;
mod config;
pub mod layers;
mod pipeline;

pub use builder::SentenceEmbeddingsBuilder;
pub use config::{
    SentenceEmbeddingsConfig, SentenceEmbeddingsModuleConfig, SentenceEmbeddingsModuleType,
    SentenceEmbeddingsModulesConfig, SentenceEmbeddingsTokenizerConfig,
};
pub use pipeline::{
    SentenceEmbeddingsModel, SentenceEmbeddingsModelOuput, SentenceEmbeddingsOption,
    SentenceEmbeddingsTokenizerOuput,
};

pub type Attention = Vec<f32>; // Length = sequence length
pub type AttentionHead = Vec<Attention>; // Length = sequence length
pub type AttentionLayer = Vec<AttentionHead>; // Length = number of heads per attention layer
pub type AttentionOutput = Vec<AttentionLayer>; // Length = number of attention layers

pub type Embedding = Vec<f32>;
