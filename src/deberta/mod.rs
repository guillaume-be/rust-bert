//! # DeBERTa :Decoding-enhanced BERT with Disentangled Attention (He et al.)
//!
//! Implementation of the DeBERTa language model ([DeBERTa :Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) He, Liu ,Gao, Chen, 2021).
//! The base model is implemented in the `deberta_model::DebertaModel` struct. Several language model heads have also been implemented, including:
//! - Question answering: `deberta_model::DebertaForQuestionAnswering`
//! - Sequence classification: `deberta_model::DebertaForSequenceClassification`
//! - Token classification (e.g. NER, POS tagging): `deberta_model::DebertaForTokenClassification`.
//!
//! # Model set-up and pre-trained weights loading
//!
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `DebertaTokenizer` using a `vocab.json` vocabulary and `merges.txt` merges file
//! Pretrained models for a number of language pairs are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::deberta::{
//!     DebertaConfig, DebertaConfigResources, DebertaForSequenceClassification,
//!     DebertaMergesResources, DebertaModelResources, DebertaVocabResources,
//! };
//! use rust_bert::resources::{RemoteResource, ResourceProvider};
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::DeBERTaTokenizer;
//!
//! let config_resource =
//!     RemoteResource::from_pretrained(DebertaConfigResources::DEBERTA_BASE_MNLI);
//! let vocab_resource = RemoteResource::from_pretrained(DebertaVocabResources::DEBERTA_BASE_MNLI);
//! let merges_resource =
//!     RemoteResource::from_pretrained(DebertaMergesResources::DEBERTA_BASE_MNLI);
//! let weights_resource =
//!     RemoteResource::from_pretrained(DebertaModelResources::DEBERTA_BASE_MNLI);
//! let config_path = config_resource.get_local_path()?;
//! let vocab_path = vocab_resource.get_local_path()?;
//! let merges_path = merges_resource.get_local_path()?;
//! let weights_path = weights_resource.get_local_path()?;
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer = DeBERTaTokenizer::from_file(
//!     vocab_path.to_str().unwrap(),
//!     merges_path.to_str().unwrap(),
//!     true,
//! )?;
//! let config = DebertaConfig::from_file(config_path);
//! let deberta_model = DebertaForSequenceClassification::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod attention;
mod deberta_model;
mod embeddings;
mod encoder;

pub use deberta_model::{
    DebertaConfig, DebertaConfigResources, DebertaForMaskedLM, DebertaForQuestionAnswering,
    DebertaForSequenceClassification, DebertaForTokenClassification, DebertaMaskedLMOutput,
    DebertaMergesResources, DebertaModel, DebertaModelResources, DebertaQuestionAnsweringOutput,
    DebertaSequenceClassificationOutput, DebertaTokenClassificationOutput, DebertaVocabResources,
};

pub(crate) use deberta_model::{
    deserialize_attention_type, x_softmax, BaseDebertaLayerNorm, ContextPooler,
    DebertaLMPredictionHead, DebertaModelOutput, PositionAttentionType, PositionAttentionTypes,
};

pub(crate) use attention::{DebertaDisentangledSelfAttention, DisentangledSelfAttention};
pub(crate) use embeddings::BaseDebertaEmbeddings;
pub(crate) use encoder::{BaseDebertaLayer, DebertaEncoderOutput};
