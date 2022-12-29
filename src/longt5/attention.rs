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

use tch::Tensor;

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
