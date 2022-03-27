//! # Ready-to-use NLP pipelines and Transformer-based models
//!
//! Rust-native state-of-the-art Natural Language Processing models and pipelines. Port of Hugging Face's [Transformers library](https://github.com/huggingface/transformers), using the [tch-rs](https://github.com/LaurentMazare/tch-rs) crate and pre-processing from [rust-tokenizers](https://github.com/guillaume-be/rust-tokenizers). Supports multi-threaded tokenization and GPU inference.
//! This repository exposes the model base architecture, task-specific heads (see below) and [ready-to-use pipelines](#ready-to-use-pipelines). [Benchmarks](#benchmarks) are available at the end of this document.
//!
//! Get started with tasks including question answering, named entity recognition, translation, summarization, text generation, conversational agents and more in just a few lines of code:
//! ```no_run
//! use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
//!
//! # fn main() -> anyhow::Result<()> {
//! let qa_model = QuestionAnsweringModel::new(Default::default())?;
//!
//! let question = String::from("Where does Amy live ?");
//! let context = String::from("Amy lives in Amsterdam");
//! let answers = qa_model.predict(&[QaInput { question, context }], 1, 32);
//! # Ok(())
//! # }
//! ```
//!
//! Output:
//! ```no_run
//! # use rust_bert::pipelines::question_answering::Answer;
//! # let output =
//! [Answer {
//!     score: 0.9976,
//!     start: 13,
//!     end: 21,
//!     answer: String::from("Amsterdam"),
//! }]
//! # ;
//! ```
//!
//! The tasks currently supported include:
//! - Translation
//! - Summarization
//! - Multi-turn dialogue
//! - Zero-shot classification
//! - Sentiment Analysis
//! - Named Entity Recognition
//! - Part of Speech tagging
//! - Question-Answering
//! - Language Generation.
//!
//! More information on these can be found in the [`pipelines` module](./pipelines/index.html)
//! - Transformer models base architectures with customized heads. These allow to load pre-trained models for customized inference in Rust
//!
//! <details>
//! <summary> <b> Click to expand to display the supported models/tasks matrix </b> </summary>
//!
//! | |**Sequence classification**|**Token classification**|**Question answering**|**Text Generation**|**Summarization**|**Translation**|**Masked LM**|
//! :-----:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:
//! DistilBERT|✅|✅|✅| | | |✅|
//! MobileBERT|✅|✅|✅| | | |✅|
//! DeBERTa|✅|✅|✅| | | |✅|
//! DeBERTa (v2)|✅|✅|✅| | | |✅|
//! FNet|✅|✅|✅| | | |✅|
//! BERT|✅|✅|✅| | | |✅|
//! RoBERTa|✅|✅|✅| | | |✅|
//! GPT| | | |✅ | | | |
//! GPT2| | | |✅ | | | |
//! GPT-Neo| | | |✅ | | | |
//! BART|✅| | |✅ |✅| | |
//! Marian| | | |  | |✅| |
//! MBart|✅| | |✅ | | | |
//! M2M100| | | |✅ | | | |
//! Electra | |✅| | | | |✅|
//! ALBERT |✅|✅|✅| | | |✅|
//! T5 | | | |✅ |✅|✅| |
//! XLNet|✅|✅|✅|✅ | | |✅|
//! Reformer|✅| |✅|✅ | | |✅|
//! ProphetNet| | | |✅ |✅ | | |
//! Longformer|✅|✅|✅| | | |✅|
//! Pegasus| | | | |✅| | |
//! </details>
//!
//! # Getting started
//!
//! This library relies on the [tch](https://github.com/LaurentMazare/tch-rs) crate for bindings to the C++ Libtorch API.
//! The libtorch library is required can be downloaded either automatically or manually. The following provides a reference on how to set-up your environment
//! to use these bindings, please refer to the [tch](https://github.com/LaurentMazare/tch-rs) for detailed information or support.
//!
//! Furthermore, this library relies on a cache folder for downloading pre-trained models.
//! This cache location defaults to `~/.cache/.rustbert`, but can be changed by setting the `RUSTBERT_CACHE` environment variable. Note that the language models used by this library are in the order of the 100s of MBs to GBs.
//!
//! ### Manual installation (recommended)
//!
//! 1. Download `libtorch` from <https://pytorch.org/get-started/locally/>. This package requires `v1.10.0`: if this version is no longer available on the "get started" page,
//! the file should be accessible by modifying the target link, for example `https://download.pytorch.org/libtorch/cu111/libtorch-shared-with-deps-1.10.0%2Bcu111.zip` for a Linux version with CUDA11.
//! 2. Extract the library to a location of your choice
//! 3. Set the following environment variables
//! ##### Linux:
//! ```bash
//! export LIBTORCH=/path/to/libtorch
//! export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
//! ```
//!
//! ##### Windows
//! ```powershell
//! $Env:LIBTORCH = "X:\path\to\libtorch"
//! $Env:Path += ";X:\path\to\libtorch\lib"
//! ```
//!
//! ### Automatic installation
//!
//! Alternatively, you can let the `build` script automatically download the `libtorch` library for you.
//! The CPU version of libtorch will be downloaded by default. To download a CUDA version, please set the environment variable `TORCH_CUDA_VERSION` to `cu111`.
//! Note that the libtorch library is large (order of several GBs for the CUDA-enabled version) and the first build may therefore take several minutes to complete.
//!
//! # Ready-to-use pipelines
//!
//! Based on Hugging Face's pipelines, ready to use end-to-end NLP pipelines are available as part of this crate. More information on these can be found in the [`pipelines` module](./pipelines/index.html)
//! The following capabilities are currently available:
//!
//! **Disclaimer**
//! The contributors of this repository are not responsible for any generation from the 3rd party utilization of the pretrained systems proposed herein.
//!
//! <details>
//! <summary> <b>1. Question Answering</b> </summary>
//!
//! Extractive question answering from a given question and context. DistilBERT model fine-tuned on SQuAD (Stanford Question Answering Dataset)
//!
//! ```no_run
//! use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
//! # fn main() -> anyhow::Result<()> {
//! let qa_model = QuestionAnsweringModel::new(Default::default())?;
//!
//! let question = String::from("Where does Amy live ?");
//! let context = String::from("Amy lives in Amsterdam");
//!
//! let answers = qa_model.predict(&[QaInput { question, context }], 1, 32);
//! # Ok(())
//! # }
//! ```
//!
//! Output: \
//! ```no_run
//! # use rust_bert::pipelines::question_answering::Answer;
//! # let output =
//! [Answer {
//!     score: 0.9976,
//!     start: 13,
//!     end: 21,
//!     answer: String::from("Amsterdam"),
//! }]
//! # ;
//! ```
//!
//! </details>
//! &nbsp;  
//! <details>
//! <summary> <b>2. Translation </b> </summary>
//!
//! Translation pipeline supporting a broad range of source and target languages. Leverages two main architectures for translation tasks:
//! - Marian-based models, for specific source/target combinations
//! - M2M100 models allowing for direct translation between 100 languages (at a higher computational cost and lower performance for some selected languages)
//!
//! Marian-based pretrained models for the following language pairs are readily available in the library - but the user can import any Pytorch-based
//! model for predictions
//! - English <-> French
//! - English <-> Spanish
//! - English <-> Portuguese
//! - English <-> Italian
//! - English <-> Catalan
//! - English <-> German
//! - English <-> Russian
//! - English <-> Chinese
//! - English <-> Dutch
//! - English <-> Swedish
//! - English <-> Arabic
//! - English <-> Hebrew
//! - English <-> Hindi
//! - French <-> German
//!
//! For languages not supported by the proposed pretrained Marian models, the user can leverage a M2M100 model supporting direct translation between 100 languages (without intermediate English translation)
//! The full list of supported languages is available in the [`pipelines` module](./pipelines/translation/enum.Language.html)
//!
//!
//! ```no_run
//! use rust_bert::pipelines::translation::{Language, TranslationModelBuilder};
//! fn main() -> anyhow::Result<()> {
//!     let model = TranslationModelBuilder::new()
//!         .with_source_languages(vec![Language::English])
//!         .with_target_languages(vec![Language::Spanish, Language::French, Language::Italian])
//!         .create_model()?;
//!     let input_text = "This is a sentence to be translated";
//!     let output = model.translate(&[input_text], None, Language::Spanish)?;
//!     for sentence in output {
//!         println!("{}", sentence);
//!     }
//!     Ok(())
//! }
//! ```
//! Output: \
//! ```no_run
//! # let output =
//! " Il s'agit d'une phrase à traduire"
//! # ;
//! ```
//!
//! </details>
//! &nbsp;  
//! <details>
//! <summary> <b>3. Summarization </b> </summary>
//!
//! Abstractive summarization using a pretrained BART model.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! # use rust_bert::pipelines::generation_utils::LanguageGenerator;
//! use rust_bert::pipelines::summarization::SummarizationModel;
//!
//! let mut model = SummarizationModel::new(Default::default())?;
//!
//! let input = ["In findings published Tuesday in Cornell University's arXiv by a team of scientists
//! from the University of Montreal and a separate report published Wednesday in Nature Astronomy by a team
//! from University College London (UCL), the presence of water vapour was confirmed in the atmosphere of K2-18b,
//! a planet circling a star in the constellation Leo. This is the first such discovery in a planet in its star's
//! habitable zone — not too hot and not too cold for liquid water to exist. The Montreal team, led by Björn Benneke,
//! used data from the NASA's Hubble telescope to assess changes in the light coming from K2-18b's star as the planet
//! passed between it and Earth. They found that certain wavelengths of light, which are usually absorbed by water,
//! weakened when the planet was in the way, indicating not only does K2-18b have an atmosphere, but the atmosphere
//! contains water in vapour form. The team from UCL then analyzed the Montreal team's data using their own software
//! and confirmed their conclusion. This was not the first time scientists have found signs of water on an exoplanet,
//! but previous discoveries were made on planets with high temperatures or other pronounced differences from Earth.
//! \"This is the first potentially habitable planet where the temperature is right and where we now know there is water,\"
//! said UCL astronomer Angelos Tsiaras. \"It's the best candidate for habitability right now.\" \"It's a good sign\",
//! said Ryan Cloutier of the Harvard–Smithsonian Center for Astrophysics, who was not one of either study's authors.
//! \"Overall,\" he continued, \"the presence of water in its atmosphere certainly improves the prospect of K2-18b being
//! a potentially habitable planet, but further observations will be required to say for sure. \"
//! K2-18b was first identified in 2015 by the Kepler space telescope. It is about 110 light-years from Earth and larger
//! but less dense. Its star, a red dwarf, is cooler than the Sun, but the planet's orbit is much closer, such that a year
//! on K2-18b lasts 33 Earth days. According to The Guardian, astronomers were optimistic that NASA's James Webb space
//! telescope — scheduled for launch in 2021 — and the European Space Agency's 2028 ARIEL program, could reveal more
//! about exoplanets like K2-18b."];
//!
//! let output = model.summarize(&input);
//! # Ok(())
//! # }
//! ```
//! (example from: [WikiNews](https://en.wikinews.org/wiki/Astronomers_find_water_vapour_in_atmosphere_of_exoplanet_K2-18b))
//!
//! Example output: \
//! ```no_run
//! # let output =
//! "Scientists have found water vapour on K2-18b, a planet 110 light-years from Earth.
//!  This is the first such discovery in a planet in its star's habitable zone.
//!  The planet is not too hot and not too cold for liquid water to exist."
//! # ;
//! ```
//!
//! </details>
//! &nbsp;  
//! <details>
//! <summary> <b>4. Dialogue Model </b> </summary>
//!
//! Conversation model based on Microsoft's [DialoGPT](https://github.com/microsoft/DialoGPT).
//! This pipeline allows the generation of single or multi-turn conversations between a human and a model.
//! The DialoGPT's page states that
//! > The human evaluation results indicate that the response generated from DialoGPT is comparable to human response quality
//! > under a single-turn conversation Turing test. ([DialoGPT repository](https://github.com/microsoft/DialoGPT))
//!
//! The model uses a `ConversationManager` to keep track of active conversations and generate responses to them.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use rust_bert::pipelines::conversation::{ConversationManager, ConversationModel};
//! let conversation_model = ConversationModel::new(Default::default())?;
//! let mut conversation_manager = ConversationManager::new();
//!
//! let conversation_id =
//!     conversation_manager.create("Going to the movies tonight - any suggestions?");
//! let output = conversation_model.generate_responses(&mut conversation_manager);
//! # Ok(())
//! # }
//! ```
//! Example output: \
//! ```no_run
//! # let output =
//! "The Big Lebowski."
//! # ;
//! ```
//!
//! </details>
//! &nbsp;  
//! <details>
//! <summary> <b>5. Natural Language Generation </b> </summary>
//!
//! Generate language based on a prompt. GPT2 and GPT available as base models.
//! Include techniques such as beam search, top-k and nucleus sampling, temperature setting and repetition penalty.
//! Supports batch generation of sentences from several prompts. Sequences will be left-padded with the model's padding token if present, the unknown token otherwise.
//! This may impact the results, it is recommended to submit prompts of similar length for best results
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use rust_bert::pipelines::text_generation::TextGenerationModel;
//! use rust_bert::pipelines::common::ModelType;
//! let mut model = TextGenerationModel::new(Default::default())?;
//! let input_context_1 = "The dog";
//! let input_context_2 = "The cat was";
//!
//! let prefix = None; // Optional prefix to append prompts with, will be excluded from the generated output
//!
//! let output = model.generate(&[input_context_1, input_context_2], prefix);
//! # Ok(())
//! # }
//! ```
//! Example output: \
//! ```no_run
//! # let output =
//! [
//!     "The dog's owners, however, did not want to be named. According to the lawsuit, the animal's owner, a 29-year",
//!     "The dog has always been part of the family. \"He was always going to be my dog and he was always looking out for me",
//!     "The dog has been able to stay in the home for more than three months now. \"It's a very good dog. She's",
//!     "The cat was discovered earlier this month in the home of a relative of the deceased. The cat\'s owner, who wished to remain anonymous,",
//!     "The cat was pulled from the street by two-year-old Jazmine.\"I didn't know what to do,\" she said",
//!     "The cat was attacked by two stray dogs and was taken to a hospital. Two other cats were also injured in the attack and are being treated."
//! ]
//! # ;
//! ```
//!
//! </details>
//! &nbsp;  
//! <details>
//! <summary> <b>6. Zero-shot classification </b> </summary>
//!
//! Performs zero-shot classification on input sentences with provided labels using a model fine-tuned for Natural Language Inference.
//! ```no_run
//! # use rust_bert::pipelines::zero_shot_classification::ZeroShotClassificationModel;
//! # fn main() -> anyhow::Result<()> {
//! let sequence_classification_model = ZeroShotClassificationModel::new(Default::default())?;
//!  let input_sentence = "Who are you voting for in 2020?";
//!  let input_sequence_2 = "The prime minister has announced a stimulus package which was widely criticized by the opposition.";
//!  let candidate_labels = &["politics", "public health", "economics", "sports"];
//!  let output = sequence_classification_model.predict_multilabel(
//!      &[input_sentence, input_sequence_2],
//!      candidate_labels,
//!      None,
//!      128,
//!  );
//! # Ok(())
//! # }
//! ```
//!
//! outputs:
//! ```no_run
//! # use rust_bert::pipelines::sequence_classification::Label;
//! let output = [
//!     [
//!         Label {
//!             text: "politics".to_string(),
//!             score: 0.972,
//!             id: 0,
//!             sentence: 0,
//!         },
//!         Label {
//!             text: "public health".to_string(),
//!             score: 0.032,
//!             id: 1,
//!             sentence: 0,
//!         },
//!         Label {
//!             text: "economics".to_string(),
//!             score: 0.006,
//!             id: 2,
//!             sentence: 0,
//!         },
//!         Label {
//!             text: "sports".to_string(),
//!             score: 0.004,
//!             id: 3,
//!             sentence: 0,
//!         },
//!     ],
//!     [
//!         Label {
//!             text: "politics".to_string(),
//!             score: 0.975,
//!             id: 0,
//!             sentence: 1,
//!         },
//!         Label {
//!             text: "economics".to_string(),
//!             score: 0.852,
//!             id: 2,
//!             sentence: 1,
//!         },
//!         Label {
//!             text: "public health".to_string(),
//!             score: 0.0818,
//!             id: 1,
//!             sentence: 1,
//!         },
//!         Label {
//!             text: "sports".to_string(),
//!             score: 0.001,
//!             id: 3,
//!             sentence: 1,
//!         },
//!     ],
//! ]
//! .to_vec();
//! ```
//!
//! </details>
//! &nbsp;  
//! <details>
//! <summary> <b>7. Sentiment analysis </b> </summary>
//!
//! Predicts the binary sentiment for a sentence. DistilBERT model fine-tuned on SST-2.
//! ```no_run
//! use rust_bert::pipelines::sentiment::SentimentModel;
//! # fn main() -> anyhow::Result<()> {
//! let sentiment_model = SentimentModel::new(Default::default())?;
//! let input = [
//!     "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
//!     "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
//!     "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
//! ];
//! let output = sentiment_model.predict(&input);
//! # Ok(())
//! # }
//! ```
//! (Example courtesy of [IMDb](http://www.imdb.com))
//!
//! Output: \
//! ```no_run
//! # use rust_bert::pipelines::sentiment::Sentiment;
//! # use rust_bert::pipelines::sentiment::SentimentPolarity::{Positive, Negative};
//! # let output =
//! [
//!     Sentiment {
//!         polarity: Positive,
//!         score: 0.998,
//!     },
//!     Sentiment {
//!         polarity: Negative,
//!         score: 0.992,
//!     },
//!     Sentiment {
//!         polarity: Positive,
//!         score: 0.999,
//!     },
//! ]
//! # ;
//! ```
//!
//! </details>
//! &nbsp;  
//! <details>
//! <summary> <b>8. Named Entity Recognition </b> </summary>
//!
//! Extracts entities (Person, Location, Organization, Miscellaneous) from text. BERT cased large model fine-tuned on CoNNL03, contributed by the [MDZ Digital Library team at the Bavarian State Library](https://github.com/dbmdz).
//! Models are currently available for English, German, Spanish and Dutch.
//! ```no_run
//! use rust_bert::pipelines::ner::NERModel;
//! # fn main() -> anyhow::Result<()> {
//! let ner_model = NERModel::new(Default::default())?;
//! let input = [
//!     "My name is Amy. I live in Paris.",
//!     "Paris is a city in France.",
//! ];
//! let output = ner_model.predict(&input);
//! # Ok(())
//! # }
//! ```
//! Output: \
//! ```no_run
//! # use rust_bert::pipelines::ner::Entity;
//! # let output =
//! [
//!     [
//!         Entity {
//!             word: String::from("Amy"),
//!             score: 0.9986,
//!             label: String::from("I-PER"),
//!         },
//!         Entity {
//!             word: String::from("Paris"),
//!             score: 0.9985,
//!             label: String::from("I-LOC"),
//!         },
//!     ],
//!     [
//!         Entity {
//!             word: String::from("Paris"),
//!             score: 0.9988,
//!             label: String::from("I-LOC"),
//!         },
//!         Entity {
//!             word: String::from("France"),
//!             score: 0.9993,
//!             label: String::from("I-LOC"),
//!         },
//!     ],
//! ]
//! # ;
//! ```
//!
//! </details>
//! &nbsp;  
//! <details>
//! <summary> <b>9. Part of Speech tagging </b> </summary>
//!
//! Extracts Part of Speech tags (Noun, Verb, Adjective...) from text.
//! ```no_run
//! use rust_bert::pipelines::pos_tagging::POSModel;
//! # fn main() -> anyhow::Result<()> {
//! let pos_model = POSModel::new(Default::default())?;
//! let input = ["My name is Bob"];
//! let output = pos_model.predict(&input);
//! # Ok(())
//! # }
//! ```
//! Output: \
//! ```no_run
//! # use rust_bert::pipelines::pos_tagging::POSTag;
//! # let output =
//! [
//!     POSTag {
//!         word: String::from("My"),
//!         score: 0.1560,
//!         label: String::from("PRP"),
//!     },
//!     POSTag {
//!         word: String::from("name"),
//!         score: 0.6565,
//!         label: String::from("NN"),
//!     },
//!     POSTag {
//!         word: String::from("is"),
//!         score: 0.3697,
//!         label: String::from("VBZ"),
//!     },
//!     POSTag {
//!         word: String::from("Bob"),
//!         score: 0.7460,
//!         label: String::from("NNP"),
//!     },
//! ]
//! # ;
//! ```
//!
//! </details>
//!
//! ## Benchmarks
//!
//! For simple pipelines (sequence classification, tokens classification, question answering) the performance between Python and Rust is expected to be comparable. This is because the most expensive part of these pipeline is the language model itself, sharing a common implementation in the Torch backend. The [End-to-end NLP Pipelines in Rust](https://www.aclweb.org/anthology/2020.nlposs-1.4/) provides a benchmarks section covering all pipelines.
//!
//! For text generation tasks (summarization, translation, conversation, free text generation), significant benefits can be expected (up to 2 to 4 times faster processing depending on the input and application). The article [Accelerating text generation with Rust](https://guillaume-be.github.io/2020-11-21/generation_benchmarks) focuses on these text generation applications and provides more details on the performance comparison to Python.
//!
//! ## Loading pretrained and custom model weights
//!
//! The base model and task-specific heads are also available for users looking to expose their own transformer based models.
//! Examples on how to prepare the date using a native tokenizers Rust library are available in `./examples` for BERT, DistilBERT, RoBERTa, GPT, GPT2 and BART.
//! Note that when importing models from Pytorch, the convention for parameters naming needs to be aligned with the Rust schema. Loading of the pre-trained weights will fail if any of the model parameters weights cannot be found in the weight files.
//! If this quality check is to be skipped, an alternative method `load_partial` can be invoked from the variables store.
//!
//! Pretrained models are available on Hugging face's [model hub](https://huggingface.co/models?filter=rust) and can be loaded using `RemoteResources` defined in this library.
//! A conversion utility script is included in `./utils` to convert Pytorch weights to a set of weights compatible with this library. This script requires Python and `torch` to be set-up, and can be used as follows:
//! `python ./utils/convert_model.py path/to/pytorch_model.bin` where `path/to/pytorch_model.bin` is the location of the original Pytorch weights.
//!
//!
//! ## Async execution
//!
//! Creating any of the models in async context will cause panics! Running extensive calculations like running predictions in a future should be avoided, too ([see here](https://docs.rs/tokio/latest/tokio/#cpu-bound-tasks-and-blocking-code)).
//!
//! It is recommended to spawn a separate thread for the models. The `async-sentiment` example displays a possible solution you could use to integrate models into async code.
//!
//!
//! ## Citation
//!
//! If you use `rust-bert` for your work, please cite [End-to-end NLP Pipelines in Rust](https://www.aclweb.org/anthology/2020.nlposs-1.4/):
//! ```bibtex
//! @inproceedings{becquin-2020-end,
//!     title = "End-to-end {NLP} Pipelines in Rust",
//!     author = "Becquin, Guillaume",
//!     booktitle = "Proceedings of Second Workshop for NLP Open Source Software (NLP-OSS)",
//!     year = "2020",
//!     publisher = "Association for Computational Linguistics",
//!     url = "https://www.aclweb.org/anthology/2020.nlposs-1.4",
//!     pages = "20--25",
//! }
//! ```
//!
//! ## Acknowledgements
//!
//! Thank you to [Hugging Face](https://huggingface.co) for hosting a set of weights compatible with this Rust library.
//! The list of ready-to-use pretrained models is listed at [https://huggingface.co/models?filter=rust](https://huggingface.co/models?filter=rust).

// These are used abundantly in this code
#![allow(clippy::assign_op_pattern, clippy::upper_case_acronyms)]

pub mod albert;
pub mod bart;
pub mod bert;
mod common;
pub mod deberta;
pub mod deberta_v2;
pub mod distilbert;
pub mod electra;
pub mod fnet;
pub mod gpt2;
pub mod gpt_neo;
pub mod longformer;
pub mod m2m_100;
pub mod marian;
pub mod mbart;
pub mod mobilebert;
pub mod openai_gpt;
pub mod pegasus;
pub mod pipelines;
pub mod prophetnet;
pub mod reformer;
pub mod roberta;
pub mod t5;
pub mod xlnet;

pub use common::error::RustBertError;
pub use common::resources;
pub use common::{Activation, Config};
