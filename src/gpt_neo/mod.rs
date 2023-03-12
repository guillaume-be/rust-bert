//! # GPT-Neo
//!
//! Implementation of the GPT-Neo language model ([The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster, Charles and Phang, Jason and He, Horace and Thite, Anish and Nabeshima, Noa and others, 2020).
//! The base model is implemented in the `gpt_neo_model::GptNeoModel` struct. A causal language modeling head is implemented in `gpt_neo_model::GptNeoForCausalLM`
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example is provided in `examples/generation_gpt_neo`, run with `cargo run --example generation_gpt_neo`.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `GPT2Tokenizer` using a `vocab.json` vocabulary and a `merges.txt` merges file
//!
//! The following pre-trained checkpoints are readily available:
//! - 125M parameters model (GptNeoModelResources::GPT_NEO_125M)
//! - 1.3B parameters model (GptNeoModelResources::GPT_NEO_1_3B)
//! - 2.7B parameters model (GptNeoModelResources::GPT_NEO_2_7B)
//!
//! ```no_run
//! use rust_bert::gpt_neo::{
//!     GptNeoConfigResources, GptNeoMergesResources, GptNeoModelResources, GptNeoVocabResources,
//! };
//! use rust_bert::pipelines::common::ModelType;
//! use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
//! use rust_bert::resources::RemoteResource;
//! use tch::Device;
//!
//! fn main() -> anyhow::Result<()> {
//!     let config_resource = Box::new(RemoteResource::from_pretrained(
//!         GptNeoConfigResources::GPT_NEO_1_3B,
//!     ));
//!     let vocab_resource = Box::new(RemoteResource::from_pretrained(
//!         GptNeoVocabResources::GPT_NEO_1_3B,
//!     ));
//!     let merges_resource = Box::new(RemoteResource::from_pretrained(
//!         GptNeoMergesResources::GPT_NEO_1_3B,
//!     ));
//!     let model_resource = Box::new(RemoteResource::from_pretrained(
//!         GptNeoModelResources::GPT_NEO_1_3B,
//!     ));
//!
//!     let text_generation_config = TextGenerationConfig {
//!         model_type: ModelType::GPTNeo,
//!         model_resource,
//!         config_resource,
//!         vocab_resource,
//!         merges_resource: Some(merges_resource),
//!         num_beams: 4,
//!         no_repeat_ngram_size: 3,
//!         device: Device::cuda_if_available(),
//!         ..Default::default()
//!     };
//!     let model = TextGenerationModel::new(text_generation_config)?;
//!
//!     let input_context_1 = "It was a very nice and sunny";
//!     let input_context_2 = "It was a gloom winter night, and";
//!     let output = model.generate(&[input_context_1, input_context_2], None);
//!
//!     for sentence in output {
//!         println!("{}", sentence);
//!     }
//!
//!     Ok(())
//! }
//! ```

mod attention;
mod decoder;
mod gpt_neo_model;

pub use gpt_neo_model::{
    GptNeoConfig, GptNeoConfigResources, GptNeoForCausalLM, GptNeoGenerator, GptNeoMergesResources,
    GptNeoModel, GptNeoModelResources, GptNeoVocabResources,
};

pub use attention::LayerState;
