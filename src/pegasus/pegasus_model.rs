// Copyright 2021, Google and The HuggingFace Inc. team. All rights reserved.
// Copyright 2021 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::bart::BartConfig;
use crate::{Activation, Config};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # Pegasus Pretrained model weight files
pub struct PegasusModelResources;

/// # Pegasus Pretrained model config files
pub struct PegasusConfigResources;

/// # Pegasus Pretrained model vocab files
pub struct PegasusVocabResources;

impl PegasusModelResources {
    /// Shared under Apache 2.0 license by the Pegasus team at https://huggingface.co/google/pegasus-cnn_dailymail. Modified with conversion to C-array format.
    pub const CNN_DAILYMAIL: (&'static str, &'static str) = (
        "pegasus-cnn_dailymail/model",
        "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/rust_model.ot",
    );
}

impl PegasusConfigResources {
    /// Shared under Apache 2.0 license by the Pegasus team at https://huggingface.co/google/pegasus-cnn_dailymail.
    pub const CNN_DAILYMAIL: (&'static str, &'static str) = (
        "pegasus-cnn_dailymail/config",
        "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/config.json",
    );
}

impl PegasusVocabResources {
    /// Shared under Apache 2.0 license by the Pegasus team at https://huggingface.co/google/pegasus-cnn_dailymail.
    pub const CNN_DAILYMAIL: (&'static str, &'static str) = (
        "pegasus-cnn_dailymail/spiece",
        "https://huggingface.co/google/pegasus-cnn_dailymail/resolve/main/spiece.model",
    );
}

/// # Pegasus model configuration
/// Defines the Pegasus model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub type PegasusConfig = BartConfig;
