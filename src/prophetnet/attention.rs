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

use tch::{Kind, Tensor};

fn compute_relative_buckets(
    num_buckets: i64,
    max_distance: i64,
    relative_positions: &Tensor,
    bidirectional: bool,
) -> Tensor {
    let inverse_relative_positions = -relative_positions;

    let (num_buckets, relative_positions_bucket, inverse_relative_positions) = if bidirectional {
        let num_buckets = num_buckets / 2;
        let relative_position_bucket =
            inverse_relative_positions.lt(0).totype(Kind::Int) * num_buckets;
        let inverse_relative_position = inverse_relative_positions.abs();
        (
            num_buckets,
            relative_position_bucket,
            inverse_relative_position,
        )
    } else {
        (
            num_buckets,
            relative_positions.zeros_like(),
            inverse_relative_positions.max1(&inverse_relative_positions.zeros_like()),
        )
    };
    let max_exact = num_buckets / 2;
    let is_small = inverse_relative_positions.lt(max_exact);
    let max_exact_f64 = max_exact as f64;
    let val_if_large = (inverse_relative_positions.totype(Kind::Float) / max_exact_f64).log()
        / (max_distance as f64 / max_exact_f64).log2()
        * (num_buckets as f64 - max_exact_f64)
        + max_exact_f64;

    let val_if_large = val_if_large
        .min1(&(val_if_large.ones_like() * (num_buckets as f64 - 1.0)))
        .totype(Kind::Int);

    let relative_positions_bucket = relative_positions_bucket
        + is_small.where1(&inverse_relative_positions.totype(Kind::Int), &val_if_large);

    relative_positions_bucket
}

fn compute_all_stream_relative_bucket(
    num_buckets: i64,
    max_distance: i64,
    position_ids: &Tensor,
) -> (Tensor, Tensor) {
    let main_stream_relative_positions =
        position_ids
            .unsqueeze(1)
            .repeat(&[1, *position_ids.size().last().unwrap(), 1])
            - position_ids.unsqueeze(-1);

    let predicting_stream_relative_positions = Tensor::cat(&[&(position_ids - 1), position_ids], 1)
        .unsqueeze(1)
        .repeat(&[1, *position_ids.size().last().unwrap(), 1])
        - position_ids.unsqueeze(-1);

    let main_relative_position_buckets = compute_relative_buckets(
        num_buckets,
        max_distance,
        &main_stream_relative_positions,
        false,
    );

    let predict_relative_position_buckets = compute_relative_buckets(
        num_buckets,
        max_distance,
        &predicting_stream_relative_positions,
        false,
    );

    (
        main_relative_position_buckets,
        predict_relative_position_buckets,
    )
}
