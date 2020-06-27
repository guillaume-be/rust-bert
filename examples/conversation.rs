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

extern crate failure;

use rust_bert::pipelines::conversation::{ConversationConfig, ConversationModel};

fn main() -> failure::Fallible<()> {
    let conversation_config = ConversationConfig {
        do_sample: false,
        ..Default::default()
    };

    let conversation_model = ConversationModel::new(conversation_config)?;

    let input = ["If you had all the money in the world, what would you buy?"];
    let history = vec![vec![]];

    let (output, history) = conversation_model.generate_responses(&input, history);

    for output in output {
        println!("{}", output);
    }

    let input = ["Where?"];

    let (output, _history) = conversation_model.generate_responses(&input, history);

    for output in output {
        println!("{}", output);
    }

    Ok(())
}
