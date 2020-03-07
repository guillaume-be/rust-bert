# rust-bert

[![Build Status](https://travis-ci.com/guillaume-be/rust-bert.svg?branch=master)](https://travis-ci.com/guillaume-be/rust-bert)
[![Latest version](https://img.shields.io/crates/v/rust_bert.svg)](https://crates.io/crates/rust_bert)
![License](https://img.shields.io/crates/l/rust_bert.svg)

Rust native BERT implementation. Port of Huggingface's [Transformers library](https://github.com/huggingface/transformers), using the [tch-rs](https://github.com/LaurentMazare/tch-rs) crate and pre-processing from [rust-tokenizers](https://https://github.com/guillaume-be/rust-tokenizers). Supports multithreaded tokenization and GPU inference.
This repository exposes the model base architecture, task-specific heads (see below) and ready-to-use pipelines.

The following models are currently implemented:

 | |**DistilBERT**|**BERT**|**RoBERTa**|**GPT**|**GPT2**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
Masked LM|✅ |✅ |✅ | | |
Sequence classification|✅ |✅ |✅| | |
Token classification|✅ |✅ | ✅| | |
Question answering|✅ |✅ |✅| | |
Multiple choices| |✅ |✅| | |
Next token prediction| | | |✅|✅|

## Ready-to-use pipelines

Based on Huggingface's pipelines, ready to use end-to-end NLP pipelines are available as part of this crate. The following capabilities are currently available:
#### 1. Question Answering
Extractive question answering from a given question and context. DistilBERT model finetuned on SQuAD (Stanford Question Answering Dataset)

```rust
    let device = Device::cuda_if_available();
    let qa_model = QuestionAnsweringModel::new(vocab_path,
                                               config_path,
                                               weights_path, device)?;
                                                        
    let question = String::from("Where does Amy live ?");
    let context = String::from("Amy lives in Amsterdam");

    let answers = qa_model.predict(QaInput { question, context }, 1, 32);
```

Output:
```
[Answer { score: 0.9976814985275269, start: 13, end: 21, answer: "Amsterdam" }]
```


#### 2. Sentiment analysis
Predicts the binary sentiment for a sentence. DistilBERT model finetuned on SST-2.
```rust
    let device = Device::cuda_if_available();
    let sentiment_classifier = SentimentClassifier::new(vocab_path,
                                                        config_path,
                                                        weights_path, device)?;
                                                        
    let input = [
        "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
        "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
        "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
    ];

    let output = sentiment_classifier.predict(input.to_vec());
```
(Example courtesy of IMDb (http://www.imdb.com))

Output:
```
[
    Sentiment { polarity: Positive, score: 0.9981985493795946 },
    Sentiment { polarity: Negative, score: 0.9927982091903687 },
    Sentiment { polarity: Positive, score: 0.9997248985164333 }
]
```

#### 3. Named Entity Recognition
Extracts entities (Person, Location, Organization, Miscellaneous) from text. BERT cased large model finetuned on CoNNL03, contributed by the [MDZ Digital Library team at the Bavarian State Library](https://github.com/dbmdz)
```rust
    let device = Device::cuda_if_available();
    let ner_model = NERModel::new(vocab_path,
                                  config_path,
                                  weights_path, device)?;

    let input = [
        "My name is Amy. I live in Paris.",
        "Paris is a city in France."
    ];
    
    let output = ner_model.predict(input.to_vec());
```
Output:
```
[
    Entity { word: "Amy", score: 0.9986, label: "I-PER" }
    Entity { word: "Paris", score: 0.9985, label: "I-LOC" }
    Entity { word: "Paris", score: 0.9988, label: "I-LOC" }
    Entity { word: "France", score: 0.9993, label: "I-LOC" }
]
```

## Base models

The base model and task-specific heads are also available for users looking to expose their own transformer based models.
Examples on how to prepare the date using a native tokenizers Rust library are available in `./examples` for BERT, DistilBERT and RoBERTa.
Note that when importing models from Pytorch, the convention for parameters naming needs to be aligned with the Rust schema. Loading of the pre-trained weights will fail if any of the model parameters weights cannot be found in the weight files.
If this quality check is to be skipped, an alternative method `load_partial` can be invoked from the variables store.

## Setup

The model configuration and vocabulary are downloaded directly from Huggingface's repository.

The model weights need to be converter to a binary format that can be read by Libtorch (the original `.bin` files are pickles and cannot be used directly). A Python script for downloading the required files & running the necessary steps is provided.

1. Compile the package: `cargo build --release`
2. Download the model files & perform necessary conversions
   - Set-up a virtual environment and install dependencies
   - run the conversion script `python /utils/download-dependencies_{MODEL_TO_DOWNLOAD}.py`. The dependencies will be downloaded to the user's home directory, under `~/rustbert/{}`
3. Run the example `cargo run --release`

