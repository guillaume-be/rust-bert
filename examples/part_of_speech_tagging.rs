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

use rust_bert::pipelines::pos_tagging::POSModel;

fn main() -> anyhow::Result<()> {
    //    Set-up model
    let pos_model = POSModel::new(Default::default())?;

    //    Define input
    let input = ["My name is Bob"];

    //    Run model
    let output = pos_model.predict(&input);
    for (pos, pos_tag) in output[0].iter().enumerate() {
        println!("{pos} - {pos_tag:?}");
    }

    Ok(())
}
