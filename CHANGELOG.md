# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.3] - 2020-02-25
### Added
- Addition of a NER pipeline
- Addition of a QuestionAnswering pipeline

### Changed
- (BREAKING) moved `SentimentClassifier` from DistilBERT module to the newly created pipelines
- Changed precision of id to label mapping of BERT config from `i32` to `i64`
- Simplified calculation of sinusoidal embeddings for DistilBERT

## [0.4.1] - 2020-02-21
### Added
- Addition of RoBERTa language model
- Addition of `BertEmbedding` trait for BERT-like models

### Changed
- Updated `BertEmbeddings` to implement the newly created `BertEmbedding` Trait
- (BREAKING) Updated `BertModel`'s embeddings to be of type `impl BertEmbedding` rather than specific embeddings, allowing to re-use the BERT structure for other models, only replacing the embeddings layer. 

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