/// Derived from https://github.com/MaartenGr/KeyBERT, shared under MIT License
///
/// Copyright (c) 2020, Maarten P. Grootendorst
/// Copyright (c) 2022, Guillaume Becquin
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in all
/// copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/// SOFTWARE.
use crate::pipelines::keywords::tokenizer::StopWordsTokenizer;
use crate::pipelines::sentence_embeddings::{
    SentenceEmbeddingsConfig, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};
use crate::RustBertError;
use regex::Regex;
use std::collections::{HashMap, HashSet};

pub enum KeywordScorerType {
    CosineSimilarity,
    MaximalMarginRelevance,
    MaxSum,
}

pub struct KeywordExtractionConfig<'a> {
    pub sentence_embeddings_config: SentenceEmbeddingsConfig,
    pub tokenizer_stopwords: Option<HashSet<&'a str>>,
    pub tokenizer_pattern: Option<Regex>,
    pub scorer_type: KeywordScorerType,
    pub num_keywords: usize,
    pub diversity: Option<f32>,
    pub max_sum_candidates: Option<usize>,
}

#[cfg(feature = "remote")]
impl Default for KeywordExtractionConfig<'_> {
    fn default() -> Self {
        let sentence_embeddings_config =
            SentenceEmbeddingsConfig::from(SentenceEmbeddingsModelType::AllMiniLmL6V2);

        Self {
            sentence_embeddings_config,
            tokenizer_stopwords: None,
            tokenizer_pattern: None,
            scorer_type: KeywordScorerType::CosineSimilarity,
            num_keywords: 5,
            diversity: None,
            max_sum_candidates: None,
        }
    }
}

pub struct KeywordExtractionModel<'a> {
    sentence_embeddings_model: SentenceEmbeddingsModel,
    tokenizer: StopWordsTokenizer<'a>,
}

impl<'a> KeywordExtractionModel<'a> {
    /// Build a new `KeywordExtractionModel`
    ///
    /// # Arguments
    ///
    /// * `config` - `KeywordExtractionConfig` object containing a sentence embeddings configuration and tokenizer-specific options
    pub fn new(
        config: KeywordExtractionConfig<'a>,
    ) -> Result<KeywordExtractionModel<'a>, RustBertError> {
        let sentence_embeddings_model =
            SentenceEmbeddingsModel::new(config.sentence_embeddings_config)?;
        let tokenizer =
            StopWordsTokenizer::new(config.tokenizer_stopwords, config.tokenizer_pattern);
        Ok(Self {
            sentence_embeddings_model,
            tokenizer,
        })
    }

    pub fn predict<S>(&self, inputs: &[S]) -> Result<(), RustBertError>
    where
        S: AsRef<str> + Sync,
    {
        let words = self.tokenizer.tokenize_list(inputs);
        let (flat_word_list, document_boundaries) =
            KeywordExtractionModel::flatten_word_list(&words);

        let document_embeddings = self.sentence_embeddings_model.encode(inputs)?;
        let word_embeddings = self.sentence_embeddings_model.encode(&flat_word_list)?;

        Ok(())
    }

    fn flatten_word_list(
        words: &[HashMap<&'a str, Vec<(usize, usize)>>],
    ) -> (Vec<&'a str>, Vec<(usize, usize)>) {
        let mut flat_word_list = Vec::new();
        let mut doc_boundaries = Vec::with_capacity(words.len());
        let mut current_index = 0;
        for doc_words_map in words {
            let doc_words = doc_words_map.keys();
            let doc_words_len = doc_words_map.len();
            flat_word_list.extend(doc_words);
            doc_boundaries.push((current_index, current_index + doc_words_len));
            current_index += doc_words_len;
        }
        (flat_word_list, doc_boundaries)
    }
}
