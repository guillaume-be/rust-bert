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

use rust_bert::longt5::{
    LongT5Config, LongT5ConfigResources, LongT5ModelResources, LongT5VocabResources,
};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::T5Tokenizer;

fn main() -> anyhow::Result<()> {
    let config_resource = Box::new(RemoteResource::from_pretrained(
        LongT5ConfigResources::TGLOBAL_BASE_BOOK_SUMMARY,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        LongT5VocabResources::TGLOBAL_BASE_BOOK_SUMMARY,
    ));
    let _weights_resource = Box::new(RemoteResource::from_pretrained(
        LongT5ModelResources::TGLOBAL_BASE_BOOK_SUMMARY,
    ));

    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    let _config = LongT5Config::from_file(config_path);
    let _tokenizer = T5Tokenizer::from_file(vocab_path.to_str().unwrap(), false)?;

    Ok(())
}
