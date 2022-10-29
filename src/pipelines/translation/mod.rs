//! # Translation pipeline
//!
//! Pipeline and utilities to perform translation from a source to a target languages. Multiple model architectures
//! (Marian, T5, MBart or M2M100) are supported offering a wide range of model size and multilingual capabilities.
//! A high number of configuration options exist, including:
//! - Model type
//! - Model resources (weights, tokenizer and configuration files)
//! - Set of source languages and target languages supported by the model (if multilingual). This should be a array-like of `Language`, with presets existing for the pretrained models registered in this library
//! - Device placement (CPU or CUDA)
//!
//! The user may provide these inputs directly by creating an adequate `TranslationConfig`, examples are provided in
//! `examples/translation_marian.rs` or `examples/translation_m2m100.rs`. A `TranslationModel` is created from the `TranslationConfig`
//! and takes input text with optional source/target languages to perform translation. Models with a single source/target language translation
//! do not require further specification. Multilingual models with multiple possible output languages require specifying the target language to translate to.
//! Models with multiple possible source language require specifying the source language for M2M100 and MBart models (and is optional for Marian models)
//!
//! ```no_run
//! use rust_bert::m2m_100::{
//!     M2M100ConfigResources, M2M100MergesResources, M2M100ModelResources, M2M100SourceLanguages,
//!     M2M100TargetLanguages, M2M100VocabResources,
//! };
//! use rust_bert::pipelines::common::ModelType;
//! use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
//! use rust_bert::resources::RemoteResource;
//! use tch::Device;
//!
//! fn main() -> anyhow::Result<()> {
//!     let model_resource = RemoteResource::from_pretrained(M2M100ModelResources::M2M100_418M);
//!     let config_resource = RemoteResource::from_pretrained(M2M100ConfigResources::M2M100_418M);
//!     let vocab_resource = RemoteResource::from_pretrained(M2M100VocabResources::M2M100_418M);
//!     let merges_resource = RemoteResource::from_pretrained(M2M100MergesResources::M2M100_418M);
//!
//!     let source_languages = M2M100SourceLanguages::M2M100_418M;
//!     let target_languages = M2M100TargetLanguages::M2M100_418M;
//!
//!     let translation_config = TranslationConfig::new(
//!         ModelType::M2M100,
//!         model_resource,
//!         config_resource,
//!         vocab_resource,
//!         Some(merges_resource),
//!         source_languages,
//!         target_languages,
//!         Device::cuda_if_available(),
//!     );
//!     let model = TranslationModel::new(translation_config)?;
//!     let source_sentence = "This sentence will be translated in multiple languages.";
//!
//!     let mut outputs = Vec::new();
//!     outputs.extend(model.translate(&[source_sentence], Language::English, Language::French)?);
//!     outputs.extend(model.translate(
//!         &[source_sentence],
//!         Language::English,
//!         Language::Spanish,
//!     )?);
//!     outputs.extend(model.translate(&[source_sentence], Language::English, Language::Hindi)?);
//!
//!     for sentence in outputs {
//!         println!("{}", sentence);
//!     }
//!     Ok(())
//! }
//! ```
//!
//! The scenario above requires the user to know the kind of model to be used for translation. In order to facilitate the selection of
//! an efficient configuration, a model builder is also available allowing to specify a flexible number of constraints and returning a
//! recommended model that fulfill the provided requirements. An example of using such a `TranslationBuilder` is given below:
//!
//! ```no_run
//! use rust_bert::pipelines::translation::{Language, TranslationModelBuilder};
//! fn main() -> anyhow::Result<()> {
//!     let model = TranslationModelBuilder::new()
//!         .with_source_languages(vec![Language::English])
//!         .with_target_languages(vec![Language::Spanish, Language::French, Language::Italian])
//!         .create_model()?;
//!
//!     let input_context_1 = "This is a sentence to be translated";
//!     let input_context_2 = "The dog did not wake up.";
//!
//!     let output =
//!         model.translate(&[input_context_1, input_context_2], None, Language::Spanish)?;
//!
//!     for sentence in output {
//!         println!("{}", sentence);
//!     }
//!     Ok(())
//! }
//! ```

mod translation_builder;
mod translation_pipeline;

pub use translation_pipeline::{Language, TranslationConfig, TranslationModel, TranslationOption};

pub use translation_builder::TranslationModelBuilder;
