// Copyright 2018 Google AI and Google Brain team.
// Copyright 2018 Carnegie Mellon University Authors.
// Copyright 2020-present, the HuggingFace Inc. team.
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

use crate::Config;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # XLNet Pretrained model weight files
pub struct XLNetModelResources;

/// # XLNet Pretrained model config files
pub struct XLNetConfigResources;

/// # XLNet Pretrained model vocab files
pub struct XLNetVocabResources;

impl XLNetModelResources {
    /// Shared under Apache 2.0 license by the XLNet Authors at https://github.com/zihangdai/xlnet. Modified with conversion to C-array format.
    pub const XLNET_BASE_V2: (&'static str, &'static str) = (
        "xlnet-base-cased/model.ot",
        "https://cdn.huggingface.co/xlnet-base-cased/rust_model.ot",
    );
}

impl XLNetConfigResources {
    /// Shared under Apache 2.0 license by the XLNet Authors at https://github.com/zihangdai/xlnet. Modified with conversion to C-array format.
    pub const XLNET_BASE_V2: (&'static str, &'static str) = (
        "xlnet-base-cased/config.json",
        "https://cdn.huggingface.co/xlnet-base-cased-config.json",
    );
}

impl XLNetVocabResources {
    /// Shared under Apache 2.0 license by the XLNet Authors at https://github.com/zihangdai/xlnet. Modified with conversion to C-array format.
    pub const XLNET_BASE_V2: (&'static str, &'static str) = (
        "xlnet-base-cased/spiece.model",
        "https://cdn.huggingface.co/xlnet-base-cased-spiece.model",
    );
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize)]
/// # Activation function used in the attention layer and masked language model head
pub enum Activation {
    /// Gaussian Error Linear Unit ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu,
    /// Rectified Linear Unit
    relu,
    /// Swish ([Ramachandran, 2017](https://arxiv.org/abs/1710.05941))
    swish,
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize)]
/// # Attention type for the model (bidirectional or unidirectional)
pub enum AttentionType {
    /// Bidirectional (XLNet)
    bi,
    /// Unidirectional (Transformer-XL)
    uni,
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize)]
/// # Summary type for the model when used for summarization
pub enum SummaryType {
    /// Hidden state stored in the last token
    last,
    /// Hidden state stored in the first token
    first,
    /// Mean of all token hidden states
    mean,
    /// Hidden state stored in the CLS token
    cls_index,
}

#[derive(Debug, Serialize, Deserialize)]
/// # XLNet model configuration
/// Defines the XLNet model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct XLNetConfig {
    pub vocab_size: i64,
    pub d_model: i64,
    pub n_layer: i64,
    pub d_head: i64,
    pub n_head: i64,
    pub d_inner: i64,
    pub ff_activation: Activation,
    pub untie_r: bool,
    pub attn_type: AttentionType,
    pub initializer_range: f32,
    pub layer_norm_eps: Option<f64>,
    pub dropout: f64,
    pub mem_len: Option<i64>,
    pub reuse_len: Option<i64>,
    pub clamp_len: Option<i64>,
    pub bi_data: bool,
    pub same_length: bool,
    pub summary_type: SummaryType,
    pub summary_use_proj: bool,
    pub summary_activation: Option<String>,
    pub summary_proj_to_labels: Option<bool>,
    pub summary_last_dropout: Option<f64>,
    pub start_n_top: Option<i64>,
    pub end_n_top: Option<i64>,
    pub use_cache: Option<bool>,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub pad_token_id: i64,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
}

impl Config<XLNetConfig> for XLNetConfig {}
