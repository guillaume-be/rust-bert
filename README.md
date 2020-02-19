# rust-bert

[![Build Status](https://travis-ci.com/guillaume-be/rust-bert.svg?branch=master)](https://travis-ci.com/guillaume-be/rust-bert)
[![Latest version](https://img.shields.io/crates/v/rust_bert.svg)](https://crates.io/crates/rust_bert)
![License](https://img.shields.io/crates/l/rust_bert.svg)

Rust native BERT implementation. Port of Huggingface's [Transformers library](https://github.com/huggingface/transformers), using the [tch-rs](https://github.com/LaurentMazare/tch-rs) crate and pre-processing from [rust-tokenizers](https://https://github.com/guillaume-be/rust-tokenizers). Supports multithreaded tokenization and GPU inference.

The following models are currently implemented:

 | |**DistilBERT**|**BERT**|**RoBERTa**
:-----:|:-----:|:-----:|:-----:
Masked LM|✅ |✅ |✅
Sequence classification|✅ |✅ |✅
Token classification|✅ |✅ | ✅
Question answering|✅ |✅ |✅
Multiple choices| |✅ |✅

An example for sentiment analysis classification is provided:

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
## Setup

The model configuration and vocabulary are downloaded directly from Huggingface's repository.

The model weights need to be converter to a binary format that can be read by Libtorch (the original `.pth` files are pickled and cannot be used directly). A Python script for downloading the required files & running the necessary steps is provided.

1. Compile the package: `cargo build --release`
2. Download the model files & perform necessary conversions
   - Set-up a virtual environment and install dependencies
   - run the conversion script `python /utils/download-dependencies.py`. The dependencies will be downloaded to the user's home directory, under `~/rustbert`
3. Run the example `cargo run --release`

