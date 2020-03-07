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
use tch::{Device, Tensor, no_grad};
use std::path::{Path, PathBuf};
use rust_tokenizers::tokenization_utils::truncate_sequences;
use std::collections::HashMap;
use std::cmp::min;
use crate::{DistilBertForQuestionAnswering, DistilBertConfig};
use tch::nn::VarStore;
use crate::common::config::Config;
use tch::kind::Kind::Float;
use std::fs;

pub struct QaInput {
    pub question: String,
    pub context: String,
}

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
    pub attention_mask: Vec<i64>,
    pub token_to_orig_map: HashMap<i64, i64>,
    pub p_mask: Vec<i8>,

}

#[derive(Debug, Clone)]
pub struct Answer {
    pub score: f64,
    pub start: usize,
    pub end: usize,
    pub answer: String,
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
    sep_idx: i64,
    max_seq_len: usize,
    doc_stride: usize,
    max_query_length: usize,
    max_answer_len: usize,
    distilbert_qa: DistilBertForQuestionAnswering,
    var_store: VarStore,
}

impl QuestionAnsweringModel {
    pub fn new(vocab_path: &Path, model_config_path: &Path, model_weight_path: &Path, device: Device)
               -> failure::Fallible<QuestionAnsweringModel> {
        let tokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), false);
        let pad_idx = *Tokenizer::vocab(&tokenizer).special_values.get("[PAD]").expect("[PAD] token not found in vocabulary");
        let sep_idx = *Tokenizer::vocab(&tokenizer).special_values.get("[SEP]").expect("[SEP] token not found in vocabulary");
        let mut var_store = VarStore::new(device);
        let mut config = DistilBertConfig::from_file(model_config_path);
//        The config for the current pre-trained question answering model indicates position embeddings which does not seem accurate
        config.sinusoidal_pos_embds = false;
        let distilbert_qa = DistilBertForQuestionAnswering::new(&var_store.root(), &config);
        var_store.load(model_weight_path)?;
        Ok(QuestionAnsweringModel {
            tokenizer,
            pad_idx,
            sep_idx,
            max_seq_len: 384,
            doc_stride: 128,
            max_query_length: 64,
            max_answer_len: 15,
            distilbert_qa,
            var_store,
        })
    }

    fn prepare_for_model(&self, question: &str, context: &str) -> (QaExample, Vec<QaFeature>, Tensor, Tensor) {
        let qa_example = QaExample::new(question, context);
        let mut input_ids: Vec<Tensor> = vec!();
        let mut attention_masks: Vec<Tensor> = vec!();

        let features = self.generate_features(&qa_example, self.max_seq_len, self.doc_stride, self.max_query_length);
        for feature in &features {
            input_ids.push(Tensor::of_slice(&feature.input_ids).to(self.var_store.device()));
            attention_masks.push(Tensor::of_slice(&feature.attention_mask).to(self.var_store.device()));
        }
        let input_ids = Tensor::stack(&input_ids, 0);
        let attention_masks = Tensor::stack(&attention_masks, 0);
        (qa_example, features, input_ids, attention_masks)
    }

    pub fn predict(&self, qa_inputs: &[QaInput], top_k: i64) -> Vec<Vec<Answer>> {
        let inputs: Vec<(QaExample, Vec<QaFeature>, Tensor, Tensor)> = qa_inputs
            .iter()
            .map(|qa_input| self.prepare_for_model(&qa_input.question, &qa_input.context))
            .collect();

        let mut all_answers = vec!();

        for (qa_example, qa_features, input_ids, attention_mask) in inputs {
            let (start_logits, end_logits, _, _) = no_grad(|| {
                self.distilbert_qa.forward_t(Some(input_ids), Some(attention_mask), None, false).unwrap()
            });
            let start_logits = start_logits.to(Device::Cpu);
            let end_logits = end_logits.to(Device::Cpu);

            let mut answers: Vec<Answer> = vec!();
            for (feature_idx, feature) in (0..start_logits.size()[0]).zip(qa_features) {
                let start = start_logits.get(feature_idx);
                let end = end_logits.get(feature_idx);
                let p_mask = (Tensor::of_slice(&feature.p_mask) - 1).abs();

                let start: Tensor = start.exp() / start.exp().sum(Float) * &p_mask;
                let end: Tensor = end.exp() / end.exp().sum(Float) * &p_mask;

                let (starts, ends, scores) = self.decode(&start, &end, top_k);

                for idx in 0..starts.len() {
                    let start_pos = feature.token_to_orig_map[&starts[idx]] as usize;
                    let end_pos = feature.token_to_orig_map[&ends[idx]] as usize;
                    let answer = qa_example.doc_tokens[start_pos..end_pos + 1].join(" ");

                    let start = qa_example.char_to_word_offset
                        .iter()
                        .position(|&v| v as usize == start_pos)
                        .unwrap();

                    let end = qa_example.char_to_word_offset
                        .iter()
                        .rposition(|&v| v as usize == end_pos)
                        .unwrap();

                    answers.push(
                        Answer { score: scores[idx], start, end, answer })
                }
            }
            answers.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            all_answers.push(answers[..(top_k as usize)].to_vec());
        }
        all_answers
    }

    fn decode(&self, start: &Tensor, end: &Tensor, top_k: i64) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        let outer = start.unsqueeze(-1).matmul(&end.unsqueeze(0));
        let start_dim = start.size()[0];
        let end_dim = end.size()[0];
        let candidates = outer.triu(0).tril(self.max_answer_len as i64 - 1).flatten(0, -1);
        let idx_sort = if top_k == 1 {
            candidates.argmax(0, true)
        } else if candidates.size()[0] < top_k {
            candidates.argsort(0, true)
        } else {
            candidates.argsort(0, true).slice(0, 0, top_k, 1)
        };

        let mut start: Vec<i64> = vec!();
        let mut end: Vec<i64> = vec!();
        let mut scores: Vec<f64> = vec!();

        for flat_index_position in 0..idx_sort.size()[0] {
            let flat_index = idx_sort.int64_value(&[flat_index_position]);
            scores.push(candidates.double_value(&[flat_index]));
            start.push(flat_index / start_dim);
            end.push(flat_index % end_dim);
        }
        (start, end, scores)
    }


    fn generate_features(&self, qa_example: &QaExample, max_seq_length: usize, doc_stride: usize, max_query_length: usize) -> Vec<QaFeature> {
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
            let (encoded_span, attention_mask) = self.encode_qa_pair(&truncated_query, &remaining_tokens, max_seq_length, doc_stride, sequence_pair_added_tokens);

            let paragraph_len = min(
                all_doc_tokens.len() - spans.len() * doc_stride,
                max_seq_length - truncated_query.len() - sequence_pair_added_tokens);

            let mut token_to_orig_map = HashMap::new();
            for i in 0..paragraph_len {
                let index = truncated_query.len() + sequence_added_tokens + i;
                token_to_orig_map.insert(index as i64, tok_to_orig_index[spans.len() * doc_stride + i] as i64);
            }

            let p_mask = self.get_mask(&encoded_span);

            let qa_feature = QaFeature { input_ids: encoded_span.token_ids, attention_mask, token_to_orig_map, p_mask };

            spans.push(qa_feature);
            if encoded_span.num_truncated_tokens == 0 {
                break;
            }
            remaining_tokens = encoded_span.overflowing_tokens
        }
        spans
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
                      sequence_pair_added_tokens: usize) -> (TokenizedInput, Vec<i64>) {
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
        let mut attention_mask = vec![1; token_ids.len()];
        if token_ids.len() < max_seq_length {
            token_ids.append(&mut vec![self.pad_idx; max_seq_length - token_ids.len()]);
            segment_ids.append(&mut vec![0; max_seq_length - segment_ids.len()]);
            attention_mask.append(&mut vec![0; max_seq_length - attention_mask.len()]);
        }
        (TokenizedInput { token_ids, segment_ids, special_tokens_mask, overflowing_tokens, num_truncated_tokens }, attention_mask)
    }

    fn get_mask(&self, encoded_span: &TokenizedInput) -> Vec<i8> {
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
        for sep_position in sep_indices {
            p_mask[sep_position] = 1;
        }
        p_mask
    }
}

pub fn squad_processor(file_path: PathBuf) -> Vec<QaInput> {
    let file = fs::File::open(file_path).expect("unable to open file");
    let json: serde_json::Value = serde_json::from_reader(file).expect("JSON not properly formatted");
    let data = json
        .get("data").expect("SQuAD file does not contain data field")
        .as_array().expect("Data array not properly formatted");

    let mut qa_inputs: Vec<QaInput> = Vec::with_capacity(data.len());
    for qa_input in data.iter() {
        let qa_input = qa_input.as_object().unwrap();
        let paragraphs = qa_input.get("paragraphs").unwrap().as_array().unwrap();
        for paragraph in paragraphs.iter() {
            let paragraph = paragraph.as_object().unwrap();
            let context = paragraph.get("context").unwrap().as_str().unwrap();
            let qas = paragraph.get("qas").unwrap().as_array().unwrap();
            for qa in qas.iter() {
                let question = qa.as_object().unwrap().get("question").unwrap().as_str().unwrap();
                qa_inputs.push(QaInput { question: question.to_owned(), context: context.to_owned() });
            }
        }
    }
    qa_inputs
}
