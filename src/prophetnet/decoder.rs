// Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
// Copyright 2020 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use tch::{Device, Kind, Tensor};

fn ngram_attention_bias(sequence_length: i64, ngram: i64, device: Device) -> Tensor {
    let left_block = Tensor::ones(
        &[ngram, sequence_length, sequence_length],
        (Kind::Float, device),
    ) * f64::NEG_INFINITY;
    let right_block = left_block.copy();
    for stream_idx in 0..ngram {
        let _ = right_block.get(stream_idx).fill_diagonal_(0, false);
        let _ = left_block.get(stream_idx).triu_(-stream_idx + 1);
    }
    let _ = left_block
        .slice(2, 0, *left_block.size().last().unwrap(), 1)
        .fill_(0);
    Tensor::cat(&[left_block, right_block], 2)
}
