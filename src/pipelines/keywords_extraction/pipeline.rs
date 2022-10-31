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
use crate::pipelines::keywords_extraction::tokenizer::StopWordsTokenizer;
use crate::pipelines::sentence_embeddings::{
    SentenceEmbeddingsConfig, SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};
use crate::RustBertError;
use regex::Regex;
use rust_tokenizers::Offset;
use std::collections::{HashMap, HashSet};
use std::mem;

#[derive(Debug, Clone)]
pub struct Keyword {
    pub text: String,
    pub score: f32,
    pub offsets: Vec<Offset>,
}

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
    scorer_type: KeywordScorerType,
    num_keywords: usize,
    diversity: Option<f32>,
    max_sum_candidates: Option<usize>,
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
            scorer_type: config.scorer_type,
            num_keywords: config.num_keywords,
            diversity: config.diversity,
            max_sum_candidates: config.max_sum_candidates,
        })
    }

    pub fn predict<S>(&self, inputs: &[S]) -> Result<Vec<Vec<Keyword>>, RustBertError>
    where
        S: AsRef<str> + Sync,
    {
        let mut words = self.tokenizer.tokenize_list(inputs);
        let (flat_word_list, document_boundaries) =
            KeywordExtractionModel::flatten_word_list(&words);

        let document_embeddings = self
            .sentence_embeddings_model
            .encode_as_tensor(inputs)?
            .embeddings;

        let word_embeddings = self
            .sentence_embeddings_model
            .encode_as_tensor(&flat_word_list)?;

        let mut output_keywords: Vec<Vec<Keyword>> = Vec::new();
        for (document_index, (start, end)) in document_boundaries.into_iter().enumerate() {
            let mut document_keywords = Vec::new();
            let document_embedding = document_embeddings.select(0, document_index as i64);
            let word_embeddings = word_embeddings
                .embeddings
                .slice(0, start as i64, end as i64, 1);
            let local_top_word_indices = self.scorer_type.score_keywords(
                document_embedding,
                word_embeddings,
                self.num_keywords,
                self.diversity,
                self.max_sum_candidates,
            );
            for (index, score) in local_top_word_indices {
                let word = flat_word_list[start + index];
                document_keywords.push(Keyword {
                    text: word.to_string(),
                    score,
                    offsets: mem::take(words[document_index].get_mut(word).unwrap()),
                });
            }
            output_keywords.push(document_keywords)
        }

        Ok(output_keywords)
    }

    fn flatten_word_list(
        words: &[HashMap<&'a str, Vec<Offset>>],
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
