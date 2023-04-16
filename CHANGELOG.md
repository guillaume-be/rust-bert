# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
## Added
- Addition of the [LongT5](https://arxiv.org/abs/2112.07916) model architecture and pretrained weights.
- Addition of `add_tokens` and `add_extra_ids` interafce methods to the `TokenizerOption`. Allow building most pipeline with custom tokenizer via `new_with_tokenizer`.

## Changed
- Bumped the tokenizers dependency from 7.x to 8.x, exposing additional options for special token mapping and adding the NLLBTokenizer.
- (BREAKING) Simplified the generation traits (removal of LMHeadModel and elimination of unnecessary specification for LanguageGenerator)
- Upgraded to `torch` 2.0 (via `tch` 0.11.0).

## Fixed
- MIN/MAX computation for float-like (was set to infinity instead of min/max)
- Remove the (unused) pooler from the set of weights for BERT Masked LM architecture

## [0.20.0] - 2023-01-21
## Added
- Addition of All-MiniLM-L6-V2 model weights
- Addition of Keyword/Keyphrases extraction pipeline based on KeyBERT (https://github.com/MaartenGr/KeyBERT)
- Addition of Masked Language Model pipeline, allowing to predict masked words.
- Support for the CodeBERT language model with pretrained models for language detection and masked token prediction

## Changed
- Addition of type aliases for the controlled generation (`PrefixAllowedFunction`) and zero-shot classification (`ZeroShotTemplate`).
- (BREAKING) `merges_resource` now optional for all pipelines.
- Allow mixing local and remote resources in pipelines.
- Upgraded to `torch` 1.13 (via `tch` 0.9.0).
- (BREAKING) Made the `max_length` argument for generation methods and pipelines optional.
- (BREAKING) Changed return type of `ModelForSequenceClassification` and `ModelForTokenClassification` to `Result<Self, RustBertError>` allowing error handling if no labels are provided in the configuration.

## Fixed
- Fixed configuration check for RoBERTa models for sentence classification.
- Fixed a bug causing the input prompt to be truncated for text generation if the prompt length was longer than `max_length`

## [0.19.0] - 2022-07-24
## Added
- Support for sentence embeddings models and pipelines, based on [SentenceTransformers](https://www.sbert.net).

## Changed
- Upgraded to `torch` 1.12 (via `tch` 0.8.0)

## Fixed
- Allow empty slices or slices of empty prompts for text generation.

## [0.18.0] - 2022-05-29
## Added
- Addition of the DeBERTa language model and support for question answering, sequence and token classification
- Addition of the DeBERTa v2/v3 language model and support for question answering, sequence and token classification
- Addition of a `new_with_tokenizer` method allowing building language model generator with a custom tokenizer (or pairing a tokenizer that was not originally designed with the model, e.g. T5 tokenizer with GPT2 model).
- (BREAKING) Addition of support for mT5 model, addition of new optional fields to T5Config
- Addition of `token_scores` field when `output_scores` is set to `true` for generation, returning the score for each token generated
- Addition of `offsets` to entities generated in the `NER` pipeline

## Changed
- (BREAKING) Updated `Resources`, moving `RemoteResource` and associated download utilities/dependencies behind a feature gate (enabled by default). Reworked the API for building and using resources. 
- Upgraded to `torch` 1.11 (via `tch` 0.7.2)
- Simplified token classification pipeline and mode aggregation now deterministic (fall back to the highest score for equally common labels)

## Fixed
- Fixed sinusoidal embeddings not being updated when loading a state dictionary (DistilBERT)

## [0.17.0] - 2021-12-19
## Changed
- Updated to `tch` 0.6.1 (libtorch 1.10)
- (BREAKING) Simplified the generics for multiple library traits taking as a rule `&[AsRef<str>]` or `&str` as inputs (no longer accepts owned types `Vec` and `String`)

## Added
- (BREAKING) Support for `bad_word_ids` generation, allowing to ban a set of word ids for all model supporting text generation
- Support for half-precision mode for all models (reducing memory footprint). A model can be converted to half-precision by calling the `half()` method on the `VarStore` is it currently stored in. Half-precision Torch kernels are not available for CPU (limited to CUDA devices)
- (BREAKING) Extension of the generation options that can be provided at runtime (after a model has been instantiated with a `GenerateConfig`), allowing to update the generation options from one text generation to another with the same model. This feature is implemented at the `LanguageGenerator` trait level, the high-level `TextGeneration` pipeline API remains unchanged.
- Addition of the FNet language model and support for sequence, token and multiple choice classification, question answering
- Addition of a full entities' prediction method supporting the IOBES scheme (merging entities token such as <New> + <York> -> <New York>)

## [0.16.0] - 2021-08-24
## Added
- (BREAKING) Support for `prefix_allowed_tokens_fn` argument for generation, allowing users to control the generation via custom functions
- (BREAKING) Support for `forced_bos_token_id` argument for generation, allowing users to force a given BOS token for generation (useful for MBart/M2M-class models)
- (BREAKING) Support for `output_scores` boolean argument for generation, allowing users to output the log-probability scores of generated sequences. Updated the return type of low-level generate API to `GeneratedTextOutput` and `GeneratedIndicesOutput` containing optional scores along with the generated output.
- Addition of the MBart Language model and support for text generation / direct translation between 50 language
- Addition of the M2M100 Language model and support for text generation / direct translation between 100 language

## Changed
- Updated GPT2 architecture to re-use embeddings for the output projection layer (resulting in smaller model weights files and memory footprint)
- Upgraded `tch` version to 0.5.0 (using `libtorch` 1.9.0)
- Changed default value of `no_repeat_ngram_size` for text generation from 3 to 0, aligning with [Python's Transformers](https://huggingface.co/transformers/main_classes/model.html?highlight=no_repeat_ngram_size#transformers.generation_utils.GenerationMixin.generate)
- Added the possibility to handle long inputs for token classification tasks (exceeding the model maximum length) using sliding windows over the input
- (BREAKING) Generalized borrowing of Tensors as input for models
- Aligned the optional `all_hidden_states` output for all models

## Fixed
- Updated T5 Decoder cross-attention to no longer use relative position bias (aligned with [Python reference update](https://github.com/huggingface/transformers/pull/8518))
- Removed hardcoded maximum length for sequence and token classification tasks, now using the model maximum position embeddings instead

## [0.15.1] - 2021-06-01
### Fixed
- Fixed conversation model panic for user inputs exceeding the maximum model length (1000 tokens)
- Fixed translation model panic for user inputs exceeding the maximum number of position embeddings

## [0.15.0] - 2021-05-16
### Added
- Addition of translation language pairs: 
  - English <-> Chinese (Simplified)
  - English <-> Chinese (Traditional)
  - English <-> Dutch
  - English <-> Swedish
  - English <-> Arabic
  - English <-> Hebrew
  - English <-> Hindi
- Addition of a Part of Speech pipeline. This pipeline allows predicting the POS tag (e.g. Noun, Adjective, Verb) of words in input sentences.
- Addition of a lightweight English Part of Speech tagging pretrained MobileBERT model
- Addition of the Pegasus language model and support for conditional generation
- Addition of a model for Pegasus summarization pretrained on the CNN-DM dataset
- Addition of the GPT-Neo language model and pretrained snapshots (125M, 1.3B and 2.7B parameters). Registration of GPT-Neo as an option for `TextGenerationPipeline`.

### Changed
- (BREAKING) Changed `classif_dropout` in `BartConfig` to be an optional field. This affects dependencies instantiating `BartConfig` from scratch, or using `classif_config` for custom model heads.
- (BREAKING) Changed token classification pipelines to return a Vec<Vec<Token>> instead of a Vec<Token>. The token-level predictions are now returned in separate vectors for each input sequence provided as an input (they were previously returned in a flattened vector)
- Simplification of the BART language model code base (also used for Marian and Pegasus language models)
- (BREAKING) Updated to `tch 0.4.1` (based on `libtorch 1.8.1`)

### Fixed
- Fixed character indexing error for Question Answering pipeline answers

### Removed
- Dependency to `itertools` crate

## [0.14.0] - 2021-02-22
### Added
- Addition of the Longformer language model, task-specific heads and registration in relevant pipelines

### Changed
- (BREAKING) Exposed additional settings for the Question Answering pipeline related to the maximum question, context and answer length. This is not backward compatible if the question answering configuration was created without using the `new` creator.
- Simplified the Question answering pipeline to rely on the offsets calculated by the tokenizers instead of a manual alignment. This results in moderate execution speed improvements for this pipeline.
- Updated the padding strategy for the Question answering pipeline. While before all sequences were padded to a fixed `max_length` (defaulting to 384), the padding is now done dynamically based on the length of the inputs. This results in a significant speed improvement for this pipeline.

### Fixed
- Fixed a bug for Question Answering for models that were not based on Wordpiece tokenization (including BPE and unigram based tokenizers). The issue was caused by the pre-tokenization step that was stripping the leading whitespace for all tokens. The performance of these models for QA should improve significantly.

## [0.13.0] - 2021-02-03
### Added
- Addition of the ProphetNet language model, task-specific heads and registration in relevant pipelines
- (BREAKING) Implementation of [Diverse Beam Search](https://arxiv.org/abs/1610.02424). This allows the generation of more diverse sequences within the number of beams. Addition of 2 new fields to the `GenerateConfig` that are propagated through all text generation configs (e.g. `TranslationConfig`): 
    - `num_beam_groups` (`Option<i64>`), indicating the number of sub-beam groups. This must be a divisor of the number of beams.
    - `diversity_penalty` (`Option<f64>`), indicating by which amount to penalize common words between beam groups. This will default to 5.5 if not provided. The impact of this diverse beam search is illustrated in the GPT2 integration tests.

### Changed
- (BREAKING) Simplified the input and output of encoder/decoder models to avoid needing to take ownership of the possibly cached encoder hidden state, offering a minor performance improvement for text generation tasks. The model output field for encoder hidden states are now optional, and only returned if the encoder hidden states were not provided for the given forward path. This may be a breaking change for low-level dependencies that manipulate directly the encoder/decoder model outputs.
- (BREAKING) Moved the language models implementation of the `PrivateLanguageGenerator` and `LanguageGenerator` traits (needed to generate text) to the model modules, cleaning up the generation_utils module.
- Updated download utilities crate, now leveraging Tokio 1.0 runtimes.

### Fixed
- Updated padding information and addition of position ids for batched GPT2 generation. Prior to this change, inputs that required padding had a lower quality for the text generated.

## [0.12.1] - 2021-01-04
### Added
- Addition of the MobileBERT language model, task-specific heads and registration in relevant pipelines

### Changed
- Made all model configurations `Clone`
- Made several base modules of the BERT language model public, and added model output `Struct` for the new publicly exposed, complex types

## [0.12.0] - 2020-11-29
### Added
- Addition of the Reformer language model, task-specific heads and registration in relevant pipelines
- Pre-trained models for DistilRoBERTa, used as a default for integration tests

### Changed
- Updated endpoint of the model resources reflecting changes to the Hugging Face's model hub
- Early stopping turned by default on for translation and summarization

## [0.11.0] - 2020-11-02
### Added
- Support for additional models for the conversational pipeline

### Changed
- Updated the version of Tokenizer crate with consistent visibility
- (BREAKING) move of teh text generation pipeline to its owned pipeline. Shared generation utilities are moved to `generation_utils`
- All models, tokenizers and pipelines are now `Send`

## [0.10.0] - 2020-10-04
### Added
- Benchmark scripts for all pipelines
- Addition of the XLNet model and task-specific heads

### Changed
- (BREAKING) Changed the download method for resources now a method of the resource itself, and leveraging the cached-path crate. 
- (BREAKING) Changed the return type of models to be output `Struct` instead of long tuples.
- (BREAKING) Changed the naming of the model main modules from `modelname` to `model_modelname` to avoid confusion with the top level module name  
- Extended the range of allowed types for pipelines input, allowing both owned `Vec` and slices, and both `String` and sting slice.
- Handling of all activations functions is mow made from a common module and `Struct`

## [0.9.0] - 2020-09-06
### Added
- Zero-shot classification pipeline using a natural language inference model

### Changed
- (BREAKING) Updated version of tokenizers crate with added options for lower casing, accent stripping and prefix addition
- Updated BART classification model to allow running their `forward` method without being mutable.

## [0.8.0] - 2020-08-25
### Added
- (BREAKING) Improved error handling via the addition of `RustBertError` and error propagation throughout the crate.

### Changed
- Updated version of tokenizers crate with improved error handling

## [0.7.12] - 2020-08-12
### Added
- Addition of the reformer language model and its integration for language generation

### Changed
- Changed model resources endpoints to leverage updated Hugging Face's model hub
- Updated the beam search processing to use vectorized operations

## [0.7.11] - 2020-07-26
### Changed
- Generalization of the accepted input for several pipelines to accept both `Vec` and slices, and to accept both `String` and `&str`

## [0.7.10] - 2020-07-08
### Added
- Addition of the ALBERT language model and task-specific heads
- Addition of German - English translation models
- Addition of the T5 language model and integration in supported pipelines (translation and summarization)

### Changed
- Updated the modules throughout the crate to accept both owned and references to varstore paths.

## [0.7.9] - 2020-06-28
### Added
- Addition of a multi-turn conversational pipeline based on DialoGPT.

## [0.7.8] - 2020-06-23
### Fixed
- Code formatting using `rustfmt`

## [0.7.7] - 2020-06-06
### Changed
- Removed the requirement for generation models to be mutable. Models are now all stateless, and no longer store an internal cache (now provided as an input).
- Updated BART model to take past layer states as an input instead of storing in internally.

### Fixed
- Fixed sequence classification model logits squeeze causing it to crash for batched inputs.

## [0.7.6] - 2020-05-27
### Added
- Addition of translation between Russian and English

### Fixed
- Fixed a bug causing downloads to be incomplete, and removes the creation of a tokio runtime for the download of resources.

## [0.7.5] - 2020-05-25
### Added
- Addition of the Marian model, leveraging a shared language model implementation with the BART model.
- Addition of translation capabilities. Supports translation between English and French, Spanish, Portuguese, Italian, Catalan and German, and between German and French.

## [0.7.4] - 2020-05-25
### Added
- Addition of multi-label classification capabilities for sequence classification via the `predict_mutilabel` function.

## [0.7.3] - 2020-05-19
### Added
- Generalization of pipelines to allow leveraging multiple model architectures. Leveraging `Enum` unpacking,  introduces `ConfigOption`, `TokenizerOption` and pipeline-specific Options.
- Addition of generic `SentenceClassificationModel` pipeline. The `SentimentModel` now leverages shared implementation for sentence classification.
- Addition of `TokenClassificationModel` pipeline. The `NERModel`now leverages shared implementation for token classification.

### Changed
- Major rework of tokenization crate, alignment with updated API

## [0.7.2] - 2020-05-03
### Fixed
- Minor bug fixes for tokenization

## [0.7.1] - 2020-05-03
### Added
- Implementation of the Electra model (generator, discriminator, task-specific heads)
- GPT2-medium and GPT2-large models

## [0.7.0] - 2020-04-26
### Added
- Addition of Resources for handling file dependencies (e.g. vocabularies, model weights, configurations). Resources may be `LocalResources` (pointing to a filesystem location) or `RemoteResources` (pointing to a remote endpoint). These resources can be passed to a `download_resource` method that returns the location in the local filesystem for both types of resources, downloading them if necessary.
- Resources specifications for all existing architectures, pointing to model files hosted on Hugging Face's model hub.

### Changed
- (BREAKING) moved the resources' specification to the `GenerateConfig` for `GPT2Generator`.
- (BREAKING) creation of pipeline configurations to contain the resources required to build the pipeline, used as an input rather than paths to local files.
- Updated the configuration for the number of target labels to use the `id2label` field instead of `num_labels` (aligning with changes in standard configuration in the Transformers library). Removed `num_labels` from configurations.
- Made the `output_attentions`, `output_hidden_states` and `torchscript` fields for DistilBERT configuration optional
- Fixed the device placement for sinusoidal embeddings for DistilBERT model.

## [0.6.2] - 2020-04-07
### Changed
- Optimization of the BART model avoiding unnecessary tensor copies for cache manipulation and residual connections.
- Optimization of DistilBERT model when embeddings are provided as an input

## [0.6.1] - 2020-04-06
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