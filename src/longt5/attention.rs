// Copyright 2022, The LongT5 Authors and HuggingFace Inc.
// Copyright 2022 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use tch::{Device, IndexOp, Kind, Tensor};

fn pad_to_multiple(x: Tensor, block_length: i64, dim: usize, pad_value: f64) -> Tensor {
    let mut x_size = x.size();
    let pad_length = -x_size[dim] % block_length;

    if x_size.iter().any(|&el| el == 0) {
        x_size[dim] += pad_length;
        Tensor::zeros(x_size.as_slice(), (x.kind(), x.device()))
    } else {
        let mut pad = vec![0i64; 2 * x.dim()];
        pad[2 * dim + 1] = pad_length;
        pad.reverse();
        x.pad(pad.as_slice(), "constant", pad_value)
    }
}

fn split_into_blocks(mut x: Tensor, block_length: i64, dim: usize) -> Tensor {
    let mut x_size = x.size();
    if x_size[dim] % block_length != 0 {
        x = pad_to_multiple(x, block_length, dim, 0f64);
    }
    let num_blocks = x_size[dim] / block_length;
    x_size.insert(dim, block_length);
    x_size.insert(dim, num_blocks);
    if x_size.iter().any(|&el| el == 0) {
        Tensor::empty(x_size.as_slice(), (x.kind(), x.device()))
    } else {
        x.reshape(x_size.as_slice())
    }
}

fn concatenate_3_blocks(
    x: &Tensor,
    block_dim: usize,
    sequence_dim: i64,
    pad_value: Option<f64>,
) -> Tensor {
    let x_size = x.size();
    let num_blocks = x_size[block_dim];
    let mut pad = vec![0i64; 2 * x.dim()];
    pad[block_dim] = 1;
    pad[block_dim + 1] = 1;
    pad.reverse();
    let x = x.pad(pad.as_slice(), "constant", pad_value.unwrap_or(0f64));
    let mut block_list: Vec<Tensor> = Vec::with_capacity(3);
    for i in 0..3 {
        block_list.push(x.narrow(block_dim as i64, i, num_blocks));
    }
    Tensor::cat(block_list.as_slice(), sequence_dim)
}

fn make_3blocks_relative_position_ids(block_length: i64, device: Device) -> Tensor {
    let position_ids = Tensor::arange(3 * block_length, (Kind::Int, device));
    let center_position_ids = position_ids.i(block_length..2 * block_length);
    position_ids.unsqueeze(0) - center_position_ids.unsqueeze(1)
}

fn mask_local_attention_mask(local_attention_mask: &Tensor, block_length: i64) -> Tensor {
    let relative_position_ids =
        make_3blocks_relative_position_ids(block_length, local_attention_mask.device());
    let locality_mask = relative_position_ids
        .abs()
        .lt(block_length)
        .unsqueeze(0)
        .unsqueeze(0);
    local_attention_mask.logical_and(&locality_mask)
}

fn get_local_attention_mask(attention_mask: Tensor, block_length: i64, device: Device) -> Tensor {
    let blocked_attention_mask = split_into_blocks(attention_mask, block_length, 1);
    let three_blocked_attention_mask = concatenate_3_blocks(&blocked_attention_mask, 1, 2, None);

    let blocked_attention_mask = blocked_attention_mask.unsqueeze(-1);
    let three_blocked_attention_mask = three_blocked_attention_mask.unsqueeze(-2);

    let local_attention_mask = mask_local_attention_mask(
        &blocked_attention_mask.logical_and(&three_blocked_attention_mask),
        block_length,
    );
    local_attention_mask.unsqueeze(1)
}
