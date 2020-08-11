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

extern crate anyhow;

use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};

fn main() -> anyhow::Result<()> {
    //    Set-up Question Answering model
    let qa_model = QuestionAnsweringModel::new(Default::default())?;

    //    Define input
    let question_1 = String::from("Where does Amy live ?");
    let context_1 = String::from("Amy lives in Amsterdam");
    let question_2 = String::from("Where does Eric live");
    let context_2 = String::from("While Amy lives in Amsterdam, Eric is in The Hague.");
    let qa_input_1 = QaInput {
        question: question_1,
        context: context_1,
    };
    let qa_input_2 = QaInput {
        question: question_2,
        context: context_2,
    };

    //    Get answer
    let answers = qa_model.predict(&vec![qa_input_1, qa_input_2], 1, 32);
    println!("{:?}", answers);
    Ok(())
}
