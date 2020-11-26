//! # XLNet (Generalized Autoregressive Pretraining for Language Understanding)
//!
//! Implementation of the XLNet language model ([Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) Yang, Dai, Yang, Carbonell, Salakhutdinov, Le, 2019).
//! The base model is implemented in the `xlnet::XLNetModel` struct. Several language model heads have also been implemented, including:
//! - Language generation: `xlnet::XLNetLMHeadModel` implementing the common `generation::LMHeadModel` trait shared between the models used for generation (see `pipelines` for more information)
//! - Multiple choices: `xlnet:XLNetForMultipleChoice`
//! - Question answering: `xlnet::XLNetForQuestionAnswering`
//! - Sequence classification: `xlnet::XLNetForSequenceClassification`
//! - Token classification (e.g. NER, POS tagging): `xlnet::XLNetForTokenClassification`.
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example (generation) is provided in `examples/generation_xlnet`, run with `cargo run --example generation_xlnet`.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `XLNetTokenizer` using a `spiece.model` sentence piece model
//!
//! Pretrained models for a number of language pairs are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use rust_bert::pipelines::common::ModelType;
//! use rust_bert::pipelines::generation_utils::{LanguageGenerator, XLNetGenerator};
//! use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
//! use rust_bert::resources::{RemoteResource, Resource};
//! use rust_bert::xlnet::{XLNetConfigResources, XLNetModelResources, XLNetVocabResources};
//! let config_resource = Resource::Remote(RemoteResource::from_pretrained(
//!     XLNetConfigResources::XLNET_BASE_CASED,
//! ));
//! let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
//!     XLNetVocabResources::XLNET_BASE_CASED,
//! ));
//! let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
//!     XLNetVocabResources::XLNET_BASE_CASED,
//! ));
//! let model_resource = Resource::Remote(RemoteResource::from_pretrained(
//!     XLNetModelResources::XLNET_BASE_CASED,
//! ));
//! let generate_config = TextGenerationConfig {
//!     model_type: ModelType::XLNet,
//!     model_resource,
//!     config_resource,
//!     vocab_resource,
//!     merges_resource,
//!     max_length: 56,
//!     do_sample: true,
//!     num_beams: 3,
//!     temperature: 1.0,
//!     num_return_sequences: 1,
//!     ..Default::default()
//! };
//! let model = TextGenerationModel::new(generate_config)?;
//! let input_context = "Once upon a time,";
//! let output = model.generate(&[input_context], None);
//!
//! # Ok(())
//! # }
//! ```

mod attention;
mod encoder;
mod xlnet_model;

pub use attention::LayerState;
pub use xlnet_model::{
    XLNetConfig, XLNetConfigResources, XLNetForMultipleChoice, XLNetForQuestionAnswering,
    XLNetForSequenceClassification, XLNetForTokenClassification, XLNetLMHeadModel, XLNetModel,
    XLNetModelResources, XLNetVocabResources,
};
