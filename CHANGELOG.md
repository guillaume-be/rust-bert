# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
## [0.7.8] - [#ToDo]
### Fixed
- Code formatting using `rustfmt`

## [0.7.7] - [#ToDo]
### Changed
- Removed the requirement for generation models to be mutable. Models are now all stateless, and no longer store an internal cache (now provided as an input).
- Updated BART model to take past layer states as an input instead of storing in internally.

### Fixed
- Fixed sequence classification model logits squeeze causing it to crash for batched inputs.

## [0.7.6] - [#ToDo]
### Added
- Addition of translation between Russian and English

### Fixed
- Fixed a bug causing downloads to be incomplete, and removes the creation of a tokio runtime for the download of resources.

## [0.7.5] - [#ToDo]
### Added
- Addition of the Marian model, leveraging a shared language model implementation with the BART model.
- Addition of translation capabilities. Supports translation between English and French, Spanish, Portuguese, Italian, Catalan and German, and between German and French.

## [0.7.4] - [#ToDo]
### Added
- Addition of multi-label classification capabilities for sequence classification via the `predict_mutilabel` function.

## [0.7.3] - [#ToDo]
### Added
- Generalization of pipelines to allow leveraging multiple model architectures. Leveraging `Enum` unpacking,  introduces `ConfigOption`, `TokenizerOption` and pipeline-specific Options.
- Addition of generic `SentenceClassificationModel` pipeline. The `SentimentModel` now leverages shared implementation for sentence classification.
- Addition of `TokenClassificationModel` pipeline. The `NERModel`now leverages shared implementation for token classification.

### Changed
- Major rework of tokenization crate, alignment with updated API

## [0.7.2] - [#ToDo]
### Fixed
- Minor bug fixes for tokenization

## [0.7.1] - [#ToDo]
### Added
- Implementation of the Electra model (generator, discriminator, task-specific heads)
- GPT2-medium and GPT2-large models

## [0.7.0] - [#ToDo]
### Added
- Addition of Resources for handling file dependencies (e.g. vocabularies, model weights, configurations). Resources may be `LocalResources` (pointing to a filesystem location) or `RemoteResources` (pointing to a remote endpoint). These resources can be passed to a `download_resource` method that returns the location in the local filesystem for both types of resources, downloading them if necessary.
- Resources specifications for all existing architectures, pointing to model files hosted on Huggingface's model hub.

### Changed
- (BREAKING) moved the resources' specification to the `GenerateConfig` for `GPT2Generator`.
- (BREAKING) creation of pipeline configurations to contain the resources required to build the pipeline, used as an input rather than paths to local files.
- Updated the configuration for the number of target labels to use the `id2label` field instead of `num_labels` (aligning with changes in standard configuration in the Transformers library). Removed `num_labels` from configurations.
- Made the `output_attentions`, `output_hidden_states` and `torchscript` fields for DistilBERT configuration optional
- Fixed the device placement for sinusoidal embeddings for DistilBERT model.


## [0.6.2] - [#ToDo]
### Changed
- Optimization of the BART model avoiding unnecessary tensor copies for cache manipulation and residual connections.
- Optimization of DistilBERT model when embeddings are provided as an input

## [0.6.1] - [#ToDo]
### Changed
- Minor optimizations to question answering and sentiment analysis pipelines
- Addition of a cache reset for text generation routines
- Implementation of cache reset for BART language model

## [0.6.0] - 2020-04-05
### Added
- BART language model
- Implementation of `LanguageModel` and `PrivateLanguageModel` for BART 
- Summarization capabilities
- Tanh activation

### Changed
- (BREAKING) Moved the `LMHeadModel` Trait from GPT2 module to the pipelines module
- Updated the `LMHeadModel` inputs to include `encoder_outputs` and `decoder_input_ids` to support causal language model (e.g. BART)
- (BREAKING) Added methods to the `PrivateLanguageGenerator` to support encoder-decoder models
- (BREAKING) changed the type of `Generator` language model to require mutability (BART caching mechanism stores the cache in the model requiring the entire model mutability - changed at a later point)
- Optimization of the `get_banned_token` method

### Fixed
- Updated the device location of the token update when EOS is not allowed because the minimum sequence length was not reached
- No longer process a given beam hypothesis if it is marked as done
- No longer add beams to a hypothesis if the rank is lower than the number of beams
- Updated final beam update to skip completed hypotheses

## [0.5.3] - 2020-03-27
### Added
- Documentation throughout the crate
- Creation of a `GenerateConfig` configuration structure to hold generation options

### Changed
- Visibility of low-level utilities in the crate
- Updated the generation options to be passed at the text generation model instantiation, rather than at every call to the `generate` method
- Updated visibility of generation routines into a public API and private lower level methods

## [0.5.2] - 2020-03-17
### Changed
- Text generation now takes a `Option<Vec<&str>>` instead of a `Option<&str>`. Shorter sequences are left-padded with `pad` if available, otherwise with `eos`.
- Turned-off gradient calculations for generation process

## [0.5.1] - 2020-03-16
### Fixed
- Beam search completion validation
- Padding sequence for sentences shorter than the maximum length moved to correct device

## [0.5.0] - 2020-03-16
### Added
- DistilGPT2 pretrained weights for GPT2
- `LMHeadModel` trait for model supporting text generation, offering an interface between the model specific input/output, and the generic set of inputs/outputs expected for model supporting text generation
- Implementation of `LMHeadModel` for GPT2 and GPT
- Text generation pipeline, supporting beam search, top-k/top-p decoding, repeated tokens banning, repetition and length penalties as `LanguageGenerator` Trait
- Implementation of `LanguageGenerator` for GPT and GPT2
- Examples and tests for language generation

### Fixed
- Fixed concatenation dimension for GPT2 past

## [0.4.5] - 2020-03-07
### Changed
- Updated input type for `QuestionAnsweringModel`'s `predict` to be `&[QaInput]` instead of a pair of question and context strings. QuestionAnsweringModel now works with a list of inputs and returns a list of predictions, processing inputs as batches.

## [0.4.4] - 2020-03-01
### Added
- Swish and gelu_new activation functions
- GPT2 language model
- GPT language model

## [0.4.3] - 2020-02-25
### Added
- Addition of a NER pipeline
- Addition of a QuestionAnswering pipeline

### Changed
- Moved `SentimentClassifier` from DistilBERT module to the newly created pipelines
- Changed precision of id to label mapping of BERT config from `i32` to `i64`
- Simplified calculation of sinusoidal embeddings for DistilBERT

## [0.4.1] - 2020-02-21
### Added
- Addition of RoBERTa language model
- Addition of `BertEmbedding` trait for BERT-like models

### Changed
- Updated `BertEmbeddings` to implement the newly created `BertEmbedding` Trait
- Updated `BertModel`'s embeddings to be of type `impl BertEmbedding` rather than specific embeddings, allowing to re-use the BERT structure for other models, only replacing the embeddings layer. 

### Fixed
- Fixed the variable path for BERT models with task-specific heads to allow loading a snapshot from models trained on Transformers.

## [0.4.0] - 2020-02-18
### Added
- BERT Model and examples
- Addition of `DistilBertForTokenClassification` and `DistilBertForQuestionAnswering` model heads
- Collection of activation functions (gelu, relu, mish)
- Dropout module
- Custom Linear layer, allowing a creation without bias
- Config trait allowing to deserialize from `json` files

### Changed
- (BREAKING) Updated `DistilBertConfig` to use the newly created `Config` Trait

## [0.3.1] - 2020-02-16
### Added
- Integration tests

### Changed
- Migrated from `rust_transformers` v0.2.0 (deprecated) to `rust_tokenizers v1.0.0

## [0.3.0] - 2020-02-13
### Added
- Example for DistilBERT masked language modeling
- Download utilities script for DistilBERT (base and SST2)

### Changed
- made `label2id`, `id2label`, `is_decoder`, `output_past` and `use_bfloat` configuration fields optional for DistilBertConfig

## [0.2.0] - 2020-02-11
### Initial release

- Tensor conversion tools from Pytorch to Libtorch format
- DistilBERT model architecture
- Ready-to-use `SentimentClassifier` using a DistilBERT model fine-tuned on SST2