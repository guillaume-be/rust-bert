//! # Sentence Embeddings pipeline
//!
//! Compute sentence/text embeddings that can be compared (e.g. with
//! cosine-similarity) to find sentences with a similar meaning. This can be useful for
//! semantic textual similar, semantic search, or paraphrase mining.
//!
//! The implementation is based on [Sentence-Transformers][sbert] and pretrained models
//! available on [Hugging Face Hub][sbert-hub] can be used. It's however necessary to
//! convert them using the script `utils/convert_model.py` beforehand, see
//! `tests/sentence_embeddings.rs` for such examples.
//!
//! [sbert]: https://sbert.net/
//! [sbert-hub]: https://huggingface.co/sentence-transformers/
//!
//! Basic usage is as follows:
//!
//! ```no_run
//! use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsBuilder;
//!
//! # fn main() -> anyhow::Result<()> {
//! let model = SentenceEmbeddingsBuilder::local("local/path/to/distiluse-base-multilingual-cased")
//!     .with_device(tch::Device::cuda_if_available())
//!     .create_model()?;
//!
//! let sentences = ["This is an example sentence", "Each sentence is converted"];
//! let embeddings = model.encode(&sentences)?;
//! # Ok(())
//! # }
//! ```

pub mod builder;
mod config;
pub mod layers;
mod pipeline;
mod resources;

pub use builder::SentenceEmbeddingsBuilder;
pub use config::{
    SentenceEmbeddingsConfig, SentenceEmbeddingsModuleConfig, SentenceEmbeddingsModuleType,
    SentenceEmbeddingsModulesConfig, SentenceEmbeddingsSentenceBertConfig,
    SentenceEmbeddingsTokenizerConfig,
};
pub use pipeline::{
    SentenceEmbeddingsModel, SentenceEmbeddingsModelOuput, SentenceEmbeddingsOption,
    SentenceEmbeddingsTokenizerOuput,
};

pub use resources::{
    SentenceEmbeddingsConfigResources, SentenceEmbeddingsDenseConfigResources,
    SentenceEmbeddingsDenseResources, SentenceEmbeddingsModelType,
    SentenceEmbeddingsModulesConfigResources, SentenceEmbeddingsPoolingConfigResources,
    SentenceEmbeddingsTokenizerConfigResources,
};

/// Length = sequence length
pub type Attention = Vec<f32>;
/// Length = sequence length
pub type AttentionHead = Vec<Attention>;
/// Length = number of heads per attention layer
pub type AttentionLayer = Vec<AttentionHead>;
/// Length = number of attention layers
pub type AttentionOutput = Vec<AttentionLayer>;

pub type Embedding = Vec<f32>;
