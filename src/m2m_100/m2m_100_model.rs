// Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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

use crate::mbart::MBartConfig;

/// # M2M100 Pretrained model weight files
pub struct M2M100ModelResources;

/// # M2M100 Pretrained model config files
pub struct M2M100ConfigResources;

/// # M2M100 Pretrained model vocab files
pub struct M2M100VocabResources;

/// # M2M100 Pretrained model ,erges files
pub struct M2M100MergesResources;

impl M2M100ModelResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const M2M100_418M: (&'static str, &'static str) = (
        "m2m100-418m/model",
        "https://huggingface.co/facebook/m2m100_418M/resolve/main/rust_model.ot",
    );
}

impl M2M100ConfigResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const M2M100_418M: (&'static str, &'static str) = (
        "m2m100-418m/config",
        "https://huggingface.co/facebook/m2m100_418M/resolve/main/config.json",
    );
}

impl M2M100VocabResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const M2M100_418M: (&'static str, &'static str) = (
        "m2m100-418m/vocab",
        "https://huggingface.co/facebook/m2m100_418M/resolve/main/vocab.json",
    );
}

impl M2M100MergesResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const M2M100_418M: (&'static str, &'static str) = (
        "m2m100-418m/merges",
        "https://huggingface.co/facebook/m2m100_418M/resolve/main/sentencepiece.bpe.model",
    );
}

pub type M2M100Config = MBartConfig;
