# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2020-04-05
### Added
- BART language model
- `LanguageModel` and `PrivateLanguageModel` implementation for BART 
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
- Fixed the variable path for BERT models with task-specific heads to allow loading snapshot from models trained on Transformers.

## [0.4.0] - 2020-02-18
### Added
- BERT Model and examples
- `DistilBertForTokenClassification` and `DistilBertForQuestionAnswering` model heads
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
- Ready-to-use `SentimentClassifier` using a DistilBERT model finetuned on SST2