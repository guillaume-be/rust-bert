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

use crate::common::activations::{Activation, TensorFunction};
use crate::common::dropout::Dropout;
use crate::xlnet::XLNetConfig;
use crate::RustBertError;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use tch::{nn, Tensor};

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
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

impl From<&XLNetConfig> for SummaryConfig {
    fn from(config: &XLNetConfig) -> Self {
        let num_labels = config
            .id2label
            .as_ref()
            .map(|id2label| id2label.len() as i64);

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
    activation: Option<TensorFunction>,
    first_dropout: Option<Dropout>,
    last_dropout: Option<Dropout>,
}

impl SequenceSummary {
    pub fn new<'p, P>(p: P, config: &SummaryConfig) -> Result<SequenceSummary, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let summary_type = config.summary_type.unwrap_or(SummaryType::last);
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
            Some(config.summary_activation.as_ref().unwrap().get_function())
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

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        cls_index: Option<&Tensor>,
        train: bool,
    ) -> Tensor {
        let mut output = match self.summary_type {
            SummaryType::last => hidden_states.select(1, -1),
            SummaryType::first => hidden_states.select(1, 0),

            SummaryType::mean => {
                hidden_states.mean_dim([1].as_slice(), false, hidden_states.kind())
            }
            SummaryType::cls_index => {
                let cls_index = if let Some(cls_index_value) = cls_index {
                    let mut expand_dim = vec![-1i64; cls_index_value.dim() - 1];
                    expand_dim.push(*hidden_states.size().last().unwrap());
                    cls_index_value
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                        .expand(expand_dim.as_slice(), true)
                } else {
                    let mut fill_value = hidden_states.size();
                    fill_value.reverse();
                    let fill_value = fill_value[2];
                    hidden_states.select(-2, 0).full_like(fill_value)
                };
                hidden_states.gather(-2, &cls_index, false).squeeze_dim(-2)
            }
        };

        if let Some(first_dropout) = &self.first_dropout {
            output = output.apply_t(first_dropout, train)
        };

        if let Some(summary) = &self.summary {
            output = output.apply(summary)
        };

        if let Some(activation_fn) = &self.activation {
            output = activation_fn.get_fn()(&output)
        };

        if let Some(last_dropout) = &self.last_dropout {
            output = output.apply_t(last_dropout, train)
        };

        output
    }
}
