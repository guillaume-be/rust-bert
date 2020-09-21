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

use crate::common::activations::{Activation, _gelu, _gelu_new, _mish, _relu, _swish, _tanh};
use crate::common::dropout::Dropout;
use crate::xlnet::XLNetConfig;
use crate::RustBertError;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use tch::{nn, Tensor};

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

pub struct SummaryConfig {
    pub summary_type: Option<SummaryType>,
    pub summary_use_proj: Option<bool>,
    pub summary_activation: Option<Activation>,
    pub summary_proj_to_labels: Option<bool>,
    pub summary_first_dropout: Option<f64>,
    pub summary_last_dropout: Option<f64>,
    pub num_labels: Option<i64>,
    pub hidden_size: i64,
}

impl From<XLNetConfig> for SummaryConfig {
    fn from(config: XLNetConfig) -> Self {
        let num_labels = if let Some(id2label) = config.id2label {
            Some(id2label.len() as i64)
        } else {
            None
        };
        SummaryConfig {
            summary_type: config.summary_type,
            summary_use_proj: config.summary_use_proj,
            summary_activation: config.summary_activation,
            summary_proj_to_labels: config.summary_proj_to_labels,
            summary_first_dropout: config.summary_first_dropout,
            summary_last_dropout: config.summary_last_dropout,
            num_labels,
            hidden_size: config.d_model,
        }
    }
}

pub struct SequenceSummary {
    summary: Option<nn::Linear>,
    summary_type: SummaryType,
    // summary_use_proj: bool,
    // summary_proj_to_labels: bool,
    activation: Option<Box<fn(&Tensor) -> Tensor>>,
    first_dropout: Option<Dropout>,
    last_dropout: Option<Dropout>,
}

impl SequenceSummary {
    pub fn new<'p, P>(p: P, config: &SummaryConfig) -> Result<SequenceSummary, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let summary_type = config.summary_type.clone().unwrap_or(SummaryType::last);
        let summary = if let Some(summary_use_proj) = config.summary_use_proj {
            let num_classes = match (config.summary_proj_to_labels, config.num_labels) {
                (Some(summary_proj_to_labels), Some(num_labels))
                    if (num_labels > 0) & summary_proj_to_labels & summary_use_proj =>
                {
                    num_labels
                }
                _ => config.hidden_size,
            };
            Some(nn::linear(
                p / "summary",
                config.hidden_size,
                num_classes,
                Default::default(),
            ))
        } else {
            None
        };

        let activation = if config.summary_activation.is_some() {
            Some(Box::new(
                match config.summary_activation.as_ref().unwrap() {
                    Activation::gelu => _gelu,
                    Activation::relu => _relu,
                    Activation::swish => _swish,
                    Activation::gelu_new => _gelu_new,
                    Activation::mish => _mish,
                    Activation::tanh => _tanh,
                },
            ))
        } else {
            None
        };

        let first_dropout = match config.summary_first_dropout {
            Some(dropout) if dropout > 0.0 => Some(Dropout::new(dropout)),
            _ => None,
        };

        let last_dropout = match config.summary_last_dropout {
            Some(dropout) if dropout > 0.0 => Some(Dropout::new(dropout)),
            _ => None,
        };

        Ok(SequenceSummary {
            summary,
            summary_type,
            activation,
            first_dropout,
            last_dropout,
        })
    }
}
