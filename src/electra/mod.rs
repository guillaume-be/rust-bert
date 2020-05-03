//! # Electra: Pre-training Text Encoders as Discriminators Rather Than Generators (Clark et al.)
//!
//! Implementation of the Electra language model ([https://openreview.net/pdf?id=r1xMH1BtvB](https://openreview.net/pdf?id=r1xMH1BtvB) Clark, Luong, Le, Manning, 2020).
//! The base model is implemented in the `electra::ElectraModel` struct. Both generator and discriminator are available via specialized heads:
//! - Generator head: `electra::ElectraGeneratorHead`
//! - Discriminator head: `electra::ElectraDiscriminatorHead`
//!
//! The generator and discriminator models are built from these:
//! - Generator (masked language model): `electra::ElectraForMaskedLM`
//! - Discriminator: `electra::ElectraDiscriminator`
//!
//! An additional sequence token classification model is available for reference
//! - Token classification (e.g. NER, POS tagging): `electra::ElectraForTokenClassification`
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example is provided in `examples/electra_masked_lm.rs`, run with `cargo run --example electra_masked_lm`.
//! The example below illustrate a Masked language model example, the structure is similar for other models (e.g. discriminator).
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `BertTokenizer` using a `vocab.txt` vocabulary
//! Pretrained models are available and can be downloaded using RemoteResources.
//!
//! ```no_run
//!# fn main() -> failure::Fallible<()> {
//!#
//! use rust_tokenizers::BertTokenizer;
//! use tch::{nn, Device};
//!# use std::path::PathBuf;
//! use rust_bert::electra::{ElectraForMaskedLM, ElectraConfig};
//! use rust_bert::Config;
//! use rust_bert::resources::{Resource, download_resource, LocalResource};
//!
//! let config_resource = Resource::Local(LocalResource { local_path: PathBuf::from("path/to/config.json")});
//! let vocab_resource = Resource::Local(LocalResource { local_path: PathBuf::from("path/to/vocab.txt")});
//! let weights_resource = Resource::Local(LocalResource { local_path: PathBuf::from("path/to/model.ot")});
//! let config_path = download_resource(&config_resource)?;
//! let vocab_path = download_resource(&vocab_resource)?;
//! let weights_path = download_resource(&weights_resource)?;
//! let device = Device::cuda_if_available();
//! let mut vs = nn::VarStore::new(device);
//! let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
//! let config = ElectraConfig::from_file(config_path);
//! let bert_model = ElectraForMaskedLM::new(&vs.root(), &config);
//! vs.load(weights_path)?;
//!
//!# Ok(())
//!# }
//! ```


mod embeddings;
mod electra;

pub use electra::{ElectraModelResources, ElectraVocabResources, ElectraConfigResources, ElectraConfig,
                  ElectraModel, ElectraDiscriminator, ElectraForMaskedLM, ElectraDiscriminatorHead, ElectraGeneratorHead, ElectraForTokenClassification};
