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


use rust_tokenizers::{BertTokenizer, Tokenizer, TruncationStrategy, TokenizedInput};
//use crate::{DistilBertForQuestionAnswering, DistilBertConfig};
//use tch::nn::VarStore;
use tch::Device;
//use crate::common::config::Config;
use std::path::Path;
use rust_tokenizers::tokenization_utils::truncate_sequences;
use std::collections::HashMap;
use std::cmp::min;

#[derive(Debug)]
pub struct QaExample {
    pub question: String,
    pub context: String,
    pub doc_tokens: Vec<String>,
    pub char_to_word_offset: Vec<i64>,
}

#[derive(Debug)]
pub struct QaFeature {
    pub input_ids: Vec<i64>,
    pub token_to_orig_map: HashMap<i64, i64>,
    pub p_mask: Vec<i8>,

}

impl QaExample {
    pub fn new(question: &str, context: &str) -> QaExample {
        let question = question.to_owned();
        let (doc_tokens, char_to_word_offset) = QaExample::split_context(context);
        QaExample { question, context: context.to_owned(), doc_tokens, char_to_word_offset }
    }

    fn split_context(context: &str) -> (Vec<String>, Vec<i64>) {
        let mut doc_tokens: Vec<String> = vec!();
        let mut char_to_word_offset: Vec<i64> = vec!();
        let max_length = context.len();
        let mut current_word = String::with_capacity(max_length);
        let mut previous_whitespace = false;

        for character in context.chars() {
            char_to_word_offset.push(doc_tokens.len() as i64);
            if QaExample::is_whitespace(&character) {
                previous_whitespace = true;
                if !current_word.is_empty() {
                    doc_tokens.push(current_word.clone());
                    current_word = String::with_capacity(max_length);
                }
            } else {
                if previous_whitespace {
                    current_word = String::with_capacity(max_length);
                }
                current_word.push(character);
                previous_whitespace = false;
            }
        }

        if !current_word.is_empty() {
            doc_tokens.push(current_word.clone());
        }
        (doc_tokens, char_to_word_offset)
    }

    fn is_whitespace(character: &char) -> bool {
        (character == &' ') |
            (character == &'\t') |
            (character == &'\r') |
            (character == &'\n') |
            (*character as u32 == 0x202F)
    }
}

pub struct QuestionAnsweringModel {
    tokenizer: BertTokenizer,
    pad_idx: i64,
    cls_idx: i64,
    sep_idx: i64,
//    _distilbert_qa: DistilBertForQuestionAnswering,
//    _var_store: VarStore,
}

impl QuestionAnsweringModel {
    pub fn new(vocab_path: &Path, _model_config_path: &Path, _model_weight_path: &Path, _device: Device)
               -> failure::Fallible<QuestionAnsweringModel> {
        let tokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), false);
        let pad_idx = *Tokenizer::vocab(&tokenizer).special_values.get("[PAD]").expect("[PAD] token not found in vocabulary");
        let cls_idx = *Tokenizer::vocab(&tokenizer).special_values.get("[CLS]").expect("[CLS] token not found in vocabulary");
        let sep_idx = *Tokenizer::vocab(&tokenizer).special_values.get("[SEP]").expect("[SEP] token not found in vocabulary");
//        let var_store = VarStore::new(device);
//        let config = DistilBertConfig::from_file(model_config_path);
//        let distilbert_qa = DistilBertForQuestionAnswering::new(&var_store.root(), &config);
//        var_store.load(model_weight_path)?;
        Ok(QuestionAnsweringModel {
            tokenizer,
            pad_idx,
            cls_idx,
            sep_idx,
//            _distilbert_qa: distilbert_qa,
//            _var_store: var_store
        })
    }

    pub fn generate_features(&self, qa_example: QaExample, max_seq_length: usize, doc_stride: usize, max_query_length: usize) {
        let mut tok_to_orig_index: Vec<i64> = vec!();
        let mut all_doc_tokens: Vec<String> = vec!();

        for (idx, token) in qa_example.doc_tokens.iter().enumerate() {
            let sub_tokens = self.tokenizer.tokenize(token);
            for sub_token in sub_tokens.into_iter() {
                all_doc_tokens.push(sub_token);
                tok_to_orig_index.push(idx as i64);
            }
        }

        let truncated_query = self.prepare_query(&qa_example.question, max_query_length);

        let sequence_added_tokens = self.tokenizer.build_input_with_special_tokens(vec!(), None).0.len();
        let sequence_pair_added_tokens = self.tokenizer.build_input_with_special_tokens(vec!(), Some(vec!())).0.len();

        let mut spans: Vec<QaFeature> = vec!();

        let mut remaining_tokens = self.tokenizer.convert_tokens_to_ids(&all_doc_tokens);
        while (spans.len() * doc_stride as usize) < all_doc_tokens.len() {
            let encoded_span = self.encode_qa_pair(&truncated_query, &remaining_tokens, max_seq_length, doc_stride, sequence_pair_added_tokens);

            let paragraph_len = min(
                all_doc_tokens.len() - spans.len() * doc_stride,
                max_seq_length - truncated_query.len() - sequence_pair_added_tokens);

            let mut token_to_orig_map = HashMap::new();
            for i in 0..paragraph_len {
                let index = truncated_query.len() + sequence_added_tokens + i;
                token_to_orig_map.insert(index as i64, tok_to_orig_index[spans.len() * doc_stride + i] as i64);
            }

            let cls_index = encoded_span.token_ids.iter().position(|v| v == &self.cls_idx).unwrap();
            let sep_indices: Vec<usize> = encoded_span.token_ids
                .iter()
                .enumerate()
                .filter(|(_, &value)| value == self.sep_idx)
                .map(|(position, _)| position)
                .collect();

            let mut p_mask: Vec<i8> = encoded_span.segment_ids
                .iter()
                .map(|v| min(v, &1i8))
                .map(|&v| 1i8 - v)
                .collect();
            p_mask[cls_index] = 0;
            for sep_position in sep_indices {
                p_mask[sep_position] = 1;
            }

            let qa_feature = QaFeature { input_ids: encoded_span.token_ids, token_to_orig_map, p_mask };

            spans.push(qa_feature);
            if encoded_span.num_truncated_tokens == 0 {
                break;
            }
            remaining_tokens = encoded_span.overflowing_tokens
        }
        println!("{:?}", spans);

        ()
    }

    fn prepare_query(&self, query: &str, max_query_length: usize) -> Vec<i64> {
        let truncated_query = self.tokenizer.convert_tokens_to_ids(&self.tokenizer.tokenize(&query));
        let num_query_tokens_to_remove = if truncated_query.len() > max_query_length as usize { truncated_query.len() - max_query_length } else { 0 };
        let (truncated_query, _, _) = truncate_sequences(truncated_query,
                                                         None,
                                                         num_query_tokens_to_remove,
                                                         &TruncationStrategy::OnlyFirst,
                                                         0).unwrap();
        truncated_query
    }

    fn encode_qa_pair(&self,
                      truncated_query: &Vec<i64>,
                      spans_token_ids: &Vec<i64>,
                      max_seq_length: usize,
                      doc_stride: usize,
                      sequence_pair_added_tokens: usize) -> TokenizedInput {
        let len_1 = truncated_query.len();
        let len_2 = spans_token_ids.len();
        let total_len = len_1 + len_2 + sequence_pair_added_tokens;
        let num_truncated_tokens = if total_len > max_seq_length { total_len - max_seq_length } else { 0 };

        let (truncated_query, truncated_context, overflowing_tokens)
            = truncate_sequences(truncated_query.clone(),
                                 Some(spans_token_ids.clone()),
                                 num_truncated_tokens,
                                 &TruncationStrategy::OnlySecond,
                                 max_seq_length - doc_stride - len_1 - sequence_pair_added_tokens).unwrap();

        let (mut token_ids, mut segment_ids, special_tokens_mask) = self.tokenizer.build_input_with_special_tokens(truncated_query, truncated_context);
        if token_ids.len() < max_seq_length {
            token_ids.append(&mut vec![self.pad_idx; max_seq_length - token_ids.len()]);
            segment_ids.append(&mut vec![0; max_seq_length - segment_ids.len()]);
        }
        TokenizedInput { token_ids, segment_ids, special_tokens_mask, overflowing_tokens, num_truncated_tokens }
    }
}












