//! # BART (Lewis et al.)
//!
//! Implementation of the BART language model ([BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) Lewis, Liu, Goyal, Ghazvininejad, Mohamed, Levy, Stoyanov, Zettlemoyer, 2019).
//! The base model is implemented in the `bart_model::BartModel` struct. The model also includes a language model head: `bart_model::BartForConditionalGeneration`
//! implementing the common `generation_utils::LMHeadModel` trait shared between the models used for generation (see `pipelines` for more information).
//!
//! # Model set-up and pre-trained weights loading
//!
//! The summarization capabilities are illustrated in `examples/summarization_bart`, run with `cargo run --example summarization_bart`.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `RobertaTokenizer` using a `vocab.txt` vocabulary and `merges.txt` 2-gram merges
//! Pretrained models are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::bart::{BartConfig, BartModel};
//! use rust_bert::resources::{LocalResource, ResourceProvider};
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::RobertaTokenizer;
//!
//! let config_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/config.json"),
//! };
//! let vocab_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/vocab.txt"),
//! };
//! let merges_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/vocab.txt"),
//! };
//! let weights_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/model.ot"),
//! };
//! let config_path = config_resource.get_local_path()?;
//! let vocab_path = vocab_resource.get_local_path()?;
//! let merges_path = merges_resource.get_local_path()?;
//! let weights_path = weights_resource.get_local_path()?;
//!
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer: RobertaTokenizer = RobertaTokenizer::from_file(
//!     vocab_path.to_str().unwrap(),
//!     merges_path.to_str().unwrap(),
//!     true,
//!     false,
//! )?;
//! let config = BartConfig::from_file(config_path);
//! let bart_model = BartModel::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod attention;
mod bart_model;
mod decoder;
mod embeddings;
mod encoder;

pub use attention::LayerState;
pub use bart_model::{
    BartConfig, BartConfigResources, BartForConditionalGeneration, BartForSequenceClassification,
    BartGenerator, BartMergesResources, BartModel, BartModelOutput, BartModelResources,
    BartVocabResources,
};

pub(crate) use attention::BartAttention;
pub(crate) use bart_model::{_expand_mask, _make_causal_mask, _prepare_decoder_attention_mask};
pub(crate) use decoder::BartDecoderOutput;
pub(crate) use embeddings::LearnedPositionalEmbedding;
pub(crate) use encoder::BartEncoderOutput;
