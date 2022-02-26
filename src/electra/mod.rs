//! # Electra: Pre-training Text Encoders as Discriminators Rather Than Generators (Clark et al.)
//!
//! Implementation of the Electra language model ([https://openreview.net/pdf?id=r1xMH1BtvB](https://openreview.net/pdf?id=r1xMH1BtvB) Clark, Luong, Le, Manning, 2020).
//! The base model is implemented in the `electra_model::ElectraModel` struct. Both generator and discriminator are available via specialized heads:
//! - Generator head: `electra_model::ElectraGeneratorHead`
//! - Discriminator head: `electra_model::ElectraDiscriminatorHead`
//!
//! The generator and discriminator models are built from these:
//! - Generator (masked language model): `electra_model::ElectraForMaskedLM`
//! - Discriminator: `electra_model::ElectraDiscriminator`
//!
//! An additional sequence token classification model is available for reference
//! - Token classification (e.g. NER, POS tagging): `electra_model::ElectraForTokenClassification`
//!
//! # Model set-up and pre-trained weights loading
//!
//! The example below illustrate a Masked language model example, the structure is similar for other models (e.g. discriminator).
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `BertTokenizer` using a `vocab.txt` vocabulary
//! Pretrained models are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! #
//! use tch::{nn, Device};
//! # use std::path::PathBuf;
//! use rust_bert::electra::{ElectraConfig, ElectraForMaskedLM};
//! use rust_bert::resources::{LocalResource, ResourceProvider};
//! use rust_bert::Config;
//! use rust_tokenizers::tokenizer::BertTokenizer;
//!
//! let config_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/config.json"),
//! };
//! let vocab_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/vocab.txt"),
//! };
//! let weights_resource = LocalResource {
//!     local_path: PathBuf::from("path/to/model.ot"),
//! };
//! let config_path = config_resource.get_local_path()?;
//! let vocab_path = vocab_resource.get_local_path()?;
//! let weights_path = weights_resource.get_local_path()?;
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer: BertTokenizer =
//!     BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
//! let config = ElectraConfig::from_file(config_path);
//! let electra_model = ElectraForMaskedLM::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//! # Ok(())
//! # }
//! ```

mod electra_model;
mod embeddings;

pub use electra_model::{
    ElectraConfig, ElectraConfigResources, ElectraDiscriminator, ElectraDiscriminatorHead,
    ElectraDiscriminatorOutput, ElectraForMaskedLM, ElectraForTokenClassification,
    ElectraGeneratorHead, ElectraMaskedLMOutput, ElectraModel, ElectraModelOutput,
    ElectraModelResources, ElectraTokenClassificationOutput, ElectraVocabResources,
};
