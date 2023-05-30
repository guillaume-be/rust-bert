//! # ONNX model support
//!
//! This crate allows running inference on models that were exported to ONNX via [onnxruntime](https://onnxruntime.ai/about.html)
//! [bindings](https://github.com/pykeio/ort). In order to use ONNX model the corresponding optional feature (`onnx`) should be turned on.
//! This will include the optional `ort` and `ndarray` dependencies. The `rust-bert` crate does not include any optional dependencies for `ort`,
//! the end user should select the set of features that would be adequate for pulling the required `onnxruntime` C++ library. The current recommended
//! installation is to use dynamic linking by pointing to an existing library location:
//! - Use the `load-dynamic` cargo feature for `ort`
//! - set the `ORT_DYLIB_PATH` to point to the location of downloaded onnxruntime library (`onnxruntime.dll`/`libonnxruntime.so`/`libonnxruntime.dylib`
//! depending on the operating system). These can be downloaded from the [release page](https://github.com/microsoft/onnxruntime/releases) of the onnxruntime project
//!
//! For troubleshooting  issues when using an ONNX model, it is recommended to add the `tracing-subscriber = { version = "0.3", default-features = false, features = [ "env-filter", "fmt" ] }`
//! dependency, and use the `tracing_subscriber::fmt::init();` instruction in the `main` binary.
//!
//! Most architectures (including encoders, decoders and encoder-decoders) are supported.
//! the library aims at keeping compatibility with models exported using the [optimum](https://github.com/huggingface/optimum) library.
//! A detailed guide on how to export a Transformer model to ONNX using optimum is available at https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model
//!
//! The resources used to create ONNX models are similar to those based on Pytorch, replacing the pytorch by the ONNX model. Since ONNX models
//! are less flexible than their Pytorch counterparts in the handling of optional arguments, exporting a decoder or encoder-decoder model to ONNX will usually
//! result in multiple files. These files are expected (but not all are necessary) for use in this library as per the table below:
//!
//! | Architecture         | Encoder file  | Decoder without past file  | Decoder with past file  |
//! |----------------------|---------------|----------------------------|-------------------------|
//! | Encoder (e.g. BERT)  | required      | not used                   | not used                |
//! | Decoder (e.g. GPT2)  | not used      | required                   | optional                |
//! | Encoder-decoder (e.g. BART)  | required      | required           | optional                |
//!
//! Note that the computational efficiency will drop when the `decoder with past` file is optional but not provided
//! since the model will not used cached past keys and values for the attention mechanism, leading to a high number of
//! redundant computations. The Optimum library offers export options to ensure such a `decoder with past` model file is created.
//!
//! The base encoder and decoder model architecture are available (and exposed for convenience) in the `encoder` and `decoder` modules, respectively.
//! Generation models (pure decoder or encoder/decoder architectures) are available in the `models` module.
//!
//! Most pipelines are available for ONNX model checkpoints, including sequence classification, zero-shot classification,
//! token classification (including named entity recognition and part-of-speech tagging), question answering, text generation, summarization and translation.
//!
//! These models use the same configuration and tokenizer files as their Pytorch counterparts when used in a pipeline. The following is
//! an example of a translation model based on a ONNX export of M2M100:
//! ```no_run
//! use rust_bert::m2m_100::{M2M100SourceLanguages, M2M100TargetLanguages};
//! use tch::Device;
//!
//! use rust_bert::pipelines::common::{ModelResource, ModelType, ONNXModelResources};
//! use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
//! use rust_bert::resources::RemoteResource;
//!
//! fn main() -> anyhow::Result<()> {
//!     let translation_model = TranslationModel::new(TranslationConfig::new(
//!         ModelType::M2M100,
//!         ModelResource::ONNX(ONNXModelResources {
//!             encoder_resource: Some(Box::new(RemoteResource::new(
//!                 "https://huggingface.co/optimum/m2m100_418M/resolve/main/encoder_model.onnx",
//!                 "onnx-m2m100_418M",
//!             ))),
//!             decoder_resource: Some(Box::new(RemoteResource::new(
//!                 "https://huggingface.co/optimum/m2m100_418M/resolve/main/decoder_model.onnx",
//!                 "onnx-m2m100_418M",
//!             ))),
//!             decoder_with_past_resource: Some(Box::new(RemoteResource::new(
//!                 "https://huggingface.co/optimum/m2m100_418M/resolve/main/decoder_with_past_model.onnx",
//!                 "onnx-m2m100_418M",
//!             ))),
//!         }),
//!         RemoteResource::new(
//!             "https://huggingface.co/optimum/m2m100_418M/resolve/main/config.json",
//!             "onnx-m2m100_418M",
//!         ),
//!         RemoteResource::new(
//!             "https://huggingface.co/optimum/m2m100_418M/resolve/main/vocab.json",
//!             "onnx-m2m100_418M",
//!         ),
//!         Some(RemoteResource::new(
//!             "https://huggingface.co/optimum/m2m100_418M/resolve/main/sentencepiece.bpe.model",
//!             "onnx-m2m100_418M",
//!         )),
//!         M2M100SourceLanguages::M2M100_418M,
//!         M2M100TargetLanguages::M2M100_418M,
//!         Device::cuda_if_available(),
//!     ))?;
//!
//!     let source_sentence = "This sentence will be translated in multiple languages.";
//!
//!     let mut outputs = Vec::new();
//!     outputs.extend(translation_model.translate(
//!         &[source_sentence],
//!         Language::English,
//!         Language::French,
//!     )?);
//!     outputs.extend(translation_model.translate(
//!         &[source_sentence],
//!         Language::English,
//!         Language::Spanish,
//!     )?);
//!     outputs.extend(translation_model.translate(
//!         &[source_sentence],
//!         Language::English,
//!         Language::Hindi,
//!     )?);
//!
//!     println!("{:?}", outputs);
//!     Ok(())
//! }
//! ```

mod common;
pub mod config;
mod conversion;
mod decoder;
mod encoder;
mod models;

pub use encoder::{ONNXEncoder, ONNXEncoderModelOutput};
pub use models::{ONNXCausalGenerator, ONNXConditionalGenerator, ONNXLayerCache, ONNXModelConfig};
