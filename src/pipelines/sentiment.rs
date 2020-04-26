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
//! use rust_bert::pipelines::sentiment::SentimentModel;
//!
//!# fn main() -> failure::Fallible<()> {
//! let sentiment_classifier = SentimentModel::new(Default::default())?;
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
use std::path::PathBuf;
use tch::{Device, Tensor, Kind, no_grad};
use tch::nn::VarStore;
use rust_tokenizers::preprocessing::tokenizer::base_tokenizer::{TruncationStrategy, MultiThreadedTokenizer};
use crate::distilbert::{DistilBertModelClassifier, DistilBertConfig, DistilBertModelResources, DistilBertConfigResources, DistilBertVocabResources};
use crate::Config;
use std::fs;
use serde::Deserialize;
use std::error::Error;
use crate::common::resources::{Resource, download_resource, RemoteResource};

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

/// # Configuration for sentiment classification
/// Contains information regarding the model to load and device to place the model on.
pub struct SentimentConfig {
    /// Model weights resource (default: pretrained DistilBERT model on SST-2)
    pub model_resource: Resource,
    /// Config resource (default: pretrained DistilBERT model on SST-2)
    pub config_resource: Resource,
    /// Vocab resource (default: pretrained DistilBERT model on SST-2)
    pub vocab_resource: Resource,
    /// Device to place the model on (default: CUDA/GPU when available)
    pub device: Device,
}

impl Default for SentimentConfig {
    fn default() -> SentimentConfig {
        SentimentConfig {
            model_resource: Resource::Remote(RemoteResource::from_pretrained(DistilBertModelResources::DISTIL_BERT_SST2)),
            config_resource: Resource::Remote(RemoteResource::from_pretrained(DistilBertConfigResources::DISTIL_BERT_SST2)),
            vocab_resource: Resource::Remote(RemoteResource::from_pretrained(DistilBertVocabResources::DISTIL_BERT_SST2)),
            device: Device::cuda_if_available(),
        }
    }
}

/// # SentimentClassifier to perform sentiment analysis
pub struct SentimentModel {
    tokenizer: BertTokenizer,
    distil_bert_classifier: DistilBertModelClassifier,
    var_store: VarStore,
}

impl SentimentModel {
    /// Build a new `SentimentModel`
    ///
    /// # Arguments
    ///
    /// * `sentiment_config` - `SentimentConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# fn main() -> failure::Fallible<()> {
    /// use rust_bert::pipelines::sentiment::SentimentModel;
    ///
    /// let sentiment_model =  SentimentModel::new(Default::default())?;
    ///# Ok(())
    ///# }
    /// ```
    ///
    pub fn new(sentiment_config: SentimentConfig) -> failure::Fallible<SentimentModel> {
        let config_path = download_resource(&sentiment_config.config_resource)?;
        let vocab_path = download_resource(&sentiment_config.vocab_resource)?;
        let weights_path = download_resource(&sentiment_config.model_resource)?;
        let device = sentiment_config.device;

        let tokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
        let mut var_store = VarStore::new(device);
        let config = DistilBertConfig::from_file(config_path);
        let distil_bert_classifier = DistilBertModelClassifier::new(&var_store.root(), &config);
        var_store.load(weights_path)?;
        Ok(SentimentModel { tokenizer, distil_bert_classifier, var_store })
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
    /// use rust_bert::pipelines::sentiment::SentimentModel;
    ///
    /// let sentiment_classifier =  SentimentModel::new(Default::default())?;
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
        let output = no_grad(|| {
            let (output, _, _) = self.distil_bert_classifier
                .forward_t(Some(input_tensor),
                           None,
                           None,
                           false)
                .unwrap();
            output.softmax(-1, Kind::Float).detach().to(Device::Cpu)
        });

        let mut sentiments: Vec<Sentiment> = vec!();
        let scores = output.select(1, 0).iter::<f64>().unwrap().collect::<Vec<f64>>();
        for score in scores {
            let polarity = if score < 0.5 { SentimentPolarity::Positive } else { SentimentPolarity::Negative };
            let score = if &SentimentPolarity::Positive == &polarity { 1.0 - score } else { score };
            sentiments.push(Sentiment { polarity, score })
        };
        sentiments
    }
}

#[derive(Debug, Deserialize)]
struct Record {
    sentence: String,
    label: i8,
}

pub fn ss2_processor(file_path: PathBuf) -> Result<Vec<String>, Box<dyn Error>> {
    let file = fs::File::open(file_path).expect("unable to open file");
    let mut csv = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b'\t')
        .from_reader(file);
    let mut records = Vec::new();
    for result in csv.deserialize() {
        let record: Record = result?;
        records.push(record.sentence);
    }
    Ok(records)
}