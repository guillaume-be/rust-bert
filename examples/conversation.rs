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

use rust_bert::pipelines::conversation::{
    ConversationConfig, ConversationManager, ConversationModel,
};

fn main() -> anyhow::Result<()> {
    let config = ConversationConfig {
        do_sample: false,
        num_beams: 3,
        ..Default::default()
    };
    let conversation_model = ConversationModel::new(config)?;
    let mut conversation_manager = ConversationManager::new();

    let conversation_1_id =
        conversation_manager.create("Going to the movies tonight - any suggestions?");
    let _conversation_2_id = conversation_manager.create("What's the last book you have read?");

    let output = conversation_model.generate_responses(&mut conversation_manager);

    println!("{output:?}");

    let _ = conversation_manager
        .get(&conversation_1_id)
        .unwrap()
        .add_user_input("Is it an action movie?");

    let output = conversation_model.generate_responses(&mut conversation_manager);

    println!("{output:?}");

    let output = conversation_model.generate_responses(&mut conversation_manager);

    println!("{output:?}");

    Ok(())
}
