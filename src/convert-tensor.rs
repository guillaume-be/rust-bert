// Copyright 2019-present, Laurent Mazare.
// Copyright 2019-present Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#[macro_use]
extern crate failure;
extern crate tch;

pub fn main() -> failure::Fallible<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 3, "usage: {} source.npz destination.ot", args[0]);

    let source_file = &args[1];
    let destination_file = &args[2];
    let tensors = tch::Tensor::read_npz(source_file)?;
    tch::Tensor::save_multi(&tensors, destination_file)?;

    Ok(())
}