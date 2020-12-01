# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [0.3.0] - 2020-02-11
### Added
- Example for DistilBERT masked language modeling
- Download utilities script for DistilBERT (base and SST2)

### Changed
- made `label2id`, `id2label`, `is_decoder`, `output_past` and `use_bfloat` configuration fields optional for DistilBertConfig

## [0.2.0] - 2020-02-11
### Initial release

- Tensor conversion tools from Pytorch to Libtorch format
- DistilBERT model architecture
- Ready-to-use sentiment classifier using a DistilBERT model finetuned on SST2