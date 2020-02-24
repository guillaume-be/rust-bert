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


#[derive(Debug)]
pub struct QaExample {
    question: String,
    context: String,
    doc_tokens: Vec<String>,
    char_to_word_offset: Vec<i64>,
}

impl QaExample {
    pub fn new(question: &str, context: &str) -> QaExample {
        let question = question.to_owned();
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

        QaExample { question, context: context.to_owned(), doc_tokens, char_to_word_offset }
    }

    fn is_whitespace(character: &char) -> bool {
        (character == &' ') |
            (character == &'\t') |
            (character == &'\r') |
            (character == &'\n') |
            (*character as u32 == 0x202F)
    }
}