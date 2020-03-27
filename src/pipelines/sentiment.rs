// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Sentiment Analysis pipeline
//! Predicts the binary sentiment for a sentence. DistilBERT model finetuned on SST-2.
//! All resources for this model can be downloaded using the Python utility script included in this repository.
//! 1. Set-up a Python virtual environment and install dependencies (in ./requirements.txt)
//! 2. Run the conversion script python /utils/download-dependencies_sst2_sentiment.py.
//! The dependencies will be downloaded to the user's home directory, under ~/rustbert/distilbert_sst2
//!
//! ```no_run
//!# use std::path::PathBuf;
//!# use tch::Device;
//! use rust_bert::pipelines::sentiment::SentimentClassifier;
//!# fn main() -> failure::Fallible<()> {
//!# let mut home: PathBuf = dirs::home_dir().unwrap();
//!# home.push("rustbert");
//!# home.push("distilbert_sst2");
//!# let config_path = &home.as_path().join("config.json");
//!# let vocab_path = &home.as_path().join("vocab.txt");
//!# let weights_path = &home.as_path().join("model.ot");
//! let device = Device::cuda_if_available();
//! let sentiment_classifier = SentimentClassifier::new(vocab_path,
//!                                                     config_path,
//!                                                     weights_path, device)?;
//! let input = [
//!     "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
//!     "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
//!     "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
//! ];
//! let output = sentiment_classifier.predict(&input);
//!# Ok(())
//!# }
//! ```
//! (Example courtesy of [IMDb](http://www.imdb.com))
//!
//! Output: \
//! ```no_run
//!# use rust_bert::pipelines::sentiment::Sentiment;
//!# use rust_bert::pipelines::sentiment::SentimentPolarity::{Positive, Negative};
//!# let output =
//! [
//!    Sentiment { polarity: Positive, score: 0.998 },
//!    Sentiment { polarity: Negative, score: 0.992 },
//!    Sentiment { polarity: Positive, score: 0.999 }
//! ]
//!# ;
//! ```

use rust_tokenizers::bert_tokenizer::BertTokenizer;
use std::path::Path;
use tch::{Device, Tensor, Kind, no_grad};
use tch::nn::VarStore;
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::{TruncationStrategy, MultiThreadedTokenizer};
use crate::distilbert::{DistilBertModelClassifier, DistilBertConfig};
use crate::Config;


#[derive(Debug, PartialEq)]
/// Enum with the possible sentiment polarities. Note that the pre-trained SST2 model does not include neutral sentiment.
pub enum SentimentPolarity {
    Positive,
    Negative,
}

#[derive(Debug)]
/// Sentiment returned by the model.
pub struct Sentiment {
    /// Polarity of the sentiment
    pub polarity: SentimentPolarity,
    /// Confidence score
    pub score: f64,
}

/// # SentimentClassifier to perform sentiment analysis
pub struct SentimentClassifier {
    tokenizer: BertTokenizer,
    distil_bert_classifier: DistilBertModelClassifier,
    var_store: VarStore,
}

impl SentimentClassifier {
    /// Build a new `SentimentClassifier`
    ///
    /// # Arguments
    ///
    /// * `vocab_path` - Path to the model vocabulary, expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers) convention
    /// * `config_path` - Path to the model configuration, expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers) convention
    /// * `weights_path` - Path to the model weight files. These need to be converted form the `.bin` to `.ot` format using the utility script provided.
    /// * `device` - Device to run the model on, e.g. `Device::Cpu` or `Device::Cuda(0)`
    ///
    /// # Example
    ///
    /// ```no_run
    ///# fn main() -> failure::Fallible<()> {
    /// use tch::Device;
    /// use std::path::{Path, PathBuf};
    /// use rust_bert::pipelines::sentiment::SentimentClassifier;
    ///
    /// let mut home: PathBuf = dirs::home_dir().unwrap();
    /// let config_path = &home.as_path().join("config.json");
    /// let vocab_path = &home.as_path().join("vocab.txt");
    /// let weights_path = &home.as_path().join("model.ot");
    /// let device = Device::Cpu;
    /// let sentiment_model =  SentimentClassifier::new(vocab_path,
    ///                                                 config_path,
    ///                                                 weights_path,
    ///                                                 device)?;
    ///# Ok(())
    ///# }
    /// ```
    ///
    pub fn new(vocab_path: &Path, config_path: &Path, weights_path: &Path, device: Device)
               -> failure::Fallible<SentimentClassifier> {
        let tokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
        let mut var_store = VarStore::new(device);
        let config = DistilBertConfig::from_file(config_path);
        let distil_bert_classifier = DistilBertModelClassifier::new(&var_store.root(), &config);
        var_store.load(weights_path)?;
        Ok(SentimentClassifier { tokenizer, distil_bert_classifier, var_store })
    }

    fn prepare_for_model(&self, input: Vec<&str>) -> Tensor {
        let tokenized_input = self.tokenizer.encode_list(input.to_vec(),
                                                         128,
                                                         &TruncationStrategy::LongestFirst,
                                                         0);
        let max_len = tokenized_input.iter().map(|input| input.token_ids.len()).max().unwrap();
        let tokenized_input = tokenized_input.
            iter().
            map(|input| input.token_ids.clone()).
            map(|mut input| {
                input.extend(vec![0; max_len - input.len()]);
                input
            }).
            map(|input|
                Tensor::of_slice(&(input))).
            collect::<Vec<_>>();
        Tensor::stack(tokenized_input.as_slice(), 0).to(self.var_store.device())
    }

    /// Extract sentiment form an array of text inputs
    ///
    /// # Arguments
    ///
    /// * `input` - `&[&str]` Array of texts to extract the sentiment from.
    ///
    /// # Returns
    /// * `Vec<Sentiment>` Sentiments extracted from texts.
    ///
    /// # Example
    ///
    /// ```no_run
    ///# fn main() -> failure::Fallible<()> {
    /// use tch::Device;
    /// use std::path::{Path, PathBuf};
    /// use rust_bert::pipelines::sentiment::SentimentClassifier;
    ///
    /// let mut home: PathBuf = dirs::home_dir().unwrap();
    /// let config_path = &home.as_path().join("config.json");
    /// let vocab_path = &home.as_path().join("vocab.txt");
    /// let weights_path = &home.as_path().join("model.ot");
    /// let device = Device::Cpu;
    /// let sentiment_classifier =  SentimentClassifier::new(vocab_path,
    ///                                                      config_path,
    ///                                                      weights_path,
    ///                                                      device)?;
    ///
    /// let input = [
    ///     "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
    ///     "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
    ///     "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
    /// ];
    ///
    /// let output = sentiment_classifier.predict(&input);
    ///# Ok(())
    ///# }
    /// ```
    ///
    pub fn predict(&self, input: &[&str]) -> Vec<Sentiment> {
        let input_tensor = self.prepare_for_model(input.to_vec());
        let (output, _, _) = no_grad(|| {
            self.distil_bert_classifier
                .forward_t(Some(input_tensor),
                           None,
                           None,
                           false)
                .unwrap()
        });
        let output = output.softmax(-1, Kind::Float);

        let mut sentiments: Vec<Sentiment> = vec!();
        for record_index in 0..output.size()[0] {
            let mut score = output.double_value(&[record_index, 0]);
            let polarity = if score < 0.5 {SentimentPolarity::Positive} else {SentimentPolarity::Negative};
            if &SentimentPolarity::Positive == &polarity {score = 1.0 - score};
            sentiments.push(Sentiment {polarity, score})
        };
        sentiments
    }
}