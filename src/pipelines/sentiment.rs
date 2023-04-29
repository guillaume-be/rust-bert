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
//! Predicts the binary sentiment for a sentence. By default, the dependencies for this
//! model will be downloaded for a DistilBERT model finetuned on SST-2.
//! Customized DistilBERT models can be loaded by overwriting the resources in the configuration.
//! The dependencies will be downloaded to the user's home directory, under ~/.cache/.rustbert/distilbert-sst2
//!
//! ```no_run
//! use rust_bert::pipelines::sentiment::SentimentModel;
//!
//! # fn main() -> anyhow::Result<()> {
//! let sentiment_classifier = SentimentModel::new(Default::default())?;
//! let input = [
//!     "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
//!     "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
//!     "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
//! ];
//! let output = sentiment_classifier.predict(&input);
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

use crate::common::error::RustBertError;
use crate::pipelines::common::TokenizerOption;
use crate::pipelines::sequence_classification::{
    SequenceClassificationConfig, SequenceClassificationModel,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
/// Enum with the possible sentiment polarities. Note that the pre-trained SST2 model does not include neutral sentiment.
pub enum SentimentPolarity {
    Positive,
    Negative,
}

#[derive(Debug, Serialize, Deserialize)]
/// Sentiment returned by the model.
pub struct Sentiment {
    /// Polarity of the sentiment
    pub polarity: SentimentPolarity,
    /// Confidence score
    pub score: f64,
}

pub type SentimentConfig = SequenceClassificationConfig;

/// # SentimentClassifier to perform sentiment analysis
pub struct SentimentModel {
    sequence_classification_model: SequenceClassificationModel,
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
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::sentiment::SentimentModel;
    ///
    /// let sentiment_model = SentimentModel::new(Default::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(sentiment_config: SentimentConfig) -> Result<SentimentModel, RustBertError> {
        let sequence_classification_model = SequenceClassificationModel::new(sentiment_config)?;
        Ok(SentimentModel {
            sequence_classification_model,
        })
    }

    /// Build a new `SentimentModel` with a provided tokenizer.
    ///
    /// # Arguments
    ///
    /// * `sentiment_config` - `SentimentConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    /// * `tokenizer` - `TokenizerOption` tokenizer to use for sentiment classification.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::common::{ModelType, TokenizerOption};
    /// use rust_bert::pipelines::sentiment::SentimentModel;
    /// let tokenizer = TokenizerOption::from_file(
    ///     ModelType::Bert,
    ///     "path/to/vocab.txt",
    ///     None,
    ///     false,
    ///     None,
    ///     None,
    /// )?;
    /// let sentiment_model = SentimentModel::new_with_tokenizer(Default::default(), tokenizer)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_tokenizer(
        sentiment_config: SentimentConfig,
        tokenizer: TokenizerOption,
    ) -> Result<SentimentModel, RustBertError> {
        let sequence_classification_model =
            SequenceClassificationModel::new_with_tokenizer(sentiment_config, tokenizer)?;
        Ok(SentimentModel {
            sequence_classification_model,
        })
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
    /// # fn main() -> anyhow::Result<()> {
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
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict<'a, S>(&self, input: S) -> Vec<Sentiment>
    where
        S: AsRef<[&'a str]>,
    {
        let labels = self.sequence_classification_model.predict(input);
        let mut sentiments = Vec::with_capacity(labels.len());
        for label in labels {
            let polarity = if label.id == 1 {
                SentimentPolarity::Positive
            } else {
                SentimentPolarity::Negative
            };
            sentiments.push(Sentiment {
                polarity,
                score: label.score,
            })
        }
        sentiments
    }
}
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[ignore] // no need to run, compilation is enough to verify it is Send
    fn test() {
        let config = SentimentConfig::default();
        let _: Box<dyn Send> = Box::new(SentimentModel::new(config));
    }
}
