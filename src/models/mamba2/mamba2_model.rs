// Copyright 2024 The HuggingFace Inc. team.
// Copyright 2024 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::common::activations::Activation;
use crate::pipelines::generation_utils::{Cache, LMModelOutput};
use crate::{Config, RustBertError};
use serde::{Deserialize, Deserializer, Serialize};
use std::borrow::Borrow;
use tch::nn::{self, embedding, Module, ModuleT};
use tch::{nn::Linear, Device, Kind, Tensor};

fn default_n_groups() -> i64 {
    8
}

/// # Mamba2 Pretrained model weight files
pub struct Mamba2ModelResources;

/// # Mamba2 Pretrained model config files
pub struct Mamba2ConfigResources;

/// # Mamba2 Pretrained model vocab files
pub struct Mamba2VocabResources;

impl Mamba2ModelResources {
    /// Shared under Apache 2.0 license by the state-spaces team at <https://huggingface.co/state-spaces/mamba2-130m>.
    pub const MAMBA2_130M: (&'static str, &'static str) = (
        "state-spaces/mamba2-130m/model",
        "https://huggingface.co/state-spaces/mamba2-130m/resolve/main/pytorch_model.bin",
    );
}

impl Mamba2ConfigResources {
    /// Shared under Apache 2.0 license by the state-spaces team at <https://huggingface.co/state-spaces/mamba2-130m>.
    pub const MAMBA2_130M: (&'static str, &'static str) = (
        "state-spaces/mamba2-130m/config",
        "https://huggingface.co/state-spaces/mamba2-130m/resolve/main/config.json",
    );
}

impl Mamba2VocabResources {
    /// Shared under Apache 2.0 license by the state-spaces team at <https://huggingface.co/state-spaces/mamba2-130m>.
    pub const MAMBA2_130M: (&'static str, &'static str) = (
        "state-spaces/mamba2-130m/tokenizer",
        "https://huggingface.co/state-spaces/mamba2-130m/resolve/main/tokenizer.json",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # Mamba2 model configuration
/// Defines the Mamba2 model architecture (e.g. number of layers, hidden layer size, state space parameters...)
pub struct Mamba2Config {
    pub num_heads: i64,
    pub head_dim: i64,
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub state_size: i64,
    pub num_hidden_layers: i64,
    pub layer_norm_epsilon: f64,
    pub pad_token_id: Option<i64>,
    pub bos_token_id: Option<i64>,
    pub eos_token_id: Option<i64>,
    pub expand: i64,
    pub conv_kernel: i64,
    #[serde(default = "default_n_groups")]
    pub n_groups: i64,
    pub use_bias: bool,
    pub use_conv_bias: bool,
    pub hidden_act: Activation,
    pub initializer_range: f64,
    pub residual_in_fp32: bool,
    pub time_step_rank: TimeStepRank,
    pub time_step_min: f64,
    pub time_step_max: f64,
    pub time_step_floor: f64,
    #[serde(deserialize_with = "deserialize_time_step_limit")]
    pub time_step_limit: (f64, f64),
    pub rescale_prenorm_residual: bool,
    pub use_cache: bool,
    pub rms_norm: bool,
    pub chunk_size: i64,
    pub tie_word_embeddings: bool,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    #[serde(default)]
    pub transformers_version: Option<String>,
    #[serde(default)]
    pub model_type: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum TimeStepRank {
    Auto(String),
    Value(i64),
}

fn deserialize_time_step_limit<'de, D>(deserializer: D) -> Result<(f64, f64), D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum FloatOrString {
        Float(f64),
        String(String),
    }

    let v: Vec<FloatOrString> = Deserialize::deserialize(deserializer)?;
    if v.len() != 2 {
        return Err(de::Error::custom("time_step_limit must have 2 elements"));
    }

    let parse_value = |val: FloatOrString| -> Result<f64, D::Error> {
        match val {
            FloatOrString::Float(f) => Ok(f),
            FloatOrString::String(s) => match s.as_str() {
                "Infinity" => Ok(f64::INFINITY),
                "-Infinity" => Ok(f64::NEG_INFINITY),
                "NaN" => Ok(f64::NAN),
                _ => Err(de::Error::custom(format!("Invalid float string: {}", s))),
            },
        }
    };

    let mut iter = v.into_iter();
    Ok((
        parse_value(iter.next().unwrap())?,
        parse_value(iter.next().unwrap())?,
    ))
}

impl Config for Mamba2Config {
    fn from_file<P: AsRef<std::path::Path>>(path: P) -> Self {
        use std::fs;
        use std::io::Read;

        let mut file = fs::File::open(path).expect("Could not open configuration file.");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Could not read configuration file.");

        // Replace JavaScript-style infinity literals with quoted strings for parsing
        // Split by lines to handle replacements more carefully
        let processed = contents
            .lines()
            .map(|line| {
                // Look for patterns like ": Infinity" or ": -Infinity" or ": NaN"
                if line.contains("Infinity") && !line.contains("\"Infinity\"") {
                    line.replace("Infinity", "\"Infinity\"")
                } else if line.contains("-Infinity") && !line.contains("\"-Infinity\"") {
                    line.replace("-Infinity", "\"-Infinity\"")
                } else if line.contains("NaN") && !line.contains("\"NaN\"") {
                    line.replace("NaN", "\"NaN\"")
                } else {
                    line.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        serde_json::from_str(&processed).expect("could not parse configuration")
    }
}

impl Default for Mamba2Config {
    fn default() -> Self {
        Mamba2Config {
            num_heads: 128,
            head_dim: 64,
            vocab_size: 32768,
            hidden_size: 4096,
            state_size: 128,
            num_hidden_layers: 64,
            layer_norm_epsilon: 1e-5,
            pad_token_id: Some(1),
            bos_token_id: Some(0),
            eos_token_id: Some(2),
            expand: 2,
            conv_kernel: 4,
            n_groups: 8,
            use_bias: false,
            use_conv_bias: true,
            hidden_act: Activation::silu,
            initializer_range: 0.1,
            residual_in_fp32: true,
            time_step_rank: TimeStepRank::Auto("auto".to_string()),
            time_step_min: 0.001,
            time_step_max: 0.1,
            time_step_floor: 1e-4,
            time_step_limit: (0.0, f64::INFINITY),
            rescale_prenorm_residual: false,
            use_cache: true,
            rms_norm: true,
            chunk_size: 256,
            tie_word_embeddings: false,
            output_attentions: None,
            output_hidden_states: None,
            transformers_version: None,
            model_type: Some("mamba2".to_string()),
        }
    }
}

impl Mamba2Config {
    fn get_time_step_rank(&self) -> i64 {
        match &self.time_step_rank {
            TimeStepRank::Auto(_) => (self.hidden_size as f64 / 16.0).ceil() as i64,
            TimeStepRank::Value(v) => *v,
        }
    }
}

/// Mamba2 Cache
#[derive(Debug)]
pub struct Mamba2Cache {
    pub conv_states: Vec<Tensor>,
    pub ssm_states: Vec<Tensor>,
}

impl Clone for Mamba2Cache {
    fn clone(&self) -> Self {
        Mamba2Cache {
            conv_states: self.conv_states.iter().map(|t| t.shallow_clone()).collect(),
            ssm_states: self.ssm_states.iter().map(|t| t.shallow_clone()).collect(),
        }
    }
}

impl Mamba2Cache {
    pub fn new(
        config: &Mamba2Config,
        num_layers: i64,
        batch_size: i64,
        device: Device,
    ) -> Mamba2Cache {
        let intermediate_size = config.expand * config.hidden_size;
        let conv_dim = intermediate_size + 2 * config.n_groups * config.state_size;

        let mut conv_states = Vec::with_capacity(num_layers as usize);
        let mut ssm_states = Vec::with_capacity(num_layers as usize);

        for _ in 0..num_layers {
            conv_states.push(Tensor::zeros(
                [batch_size, conv_dim, config.conv_kernel],
                (Kind::Float, device),
            ));
            ssm_states.push(Tensor::zeros(
                [
                    batch_size,
                    config.num_heads,
                    config.head_dim,
                    config.state_size,
                ],
                (Kind::Float, device),
            ));
        }

        Mamba2Cache {
            conv_states,
            ssm_states,
        }
    }
}

/// Mamba2 RMS Norm with optional gating
struct MambaRMSNormGated {
    weight: Tensor,
    variance_epsilon: f64,
}

impl MambaRMSNormGated {
    fn new<'p, P: Borrow<nn::Path<'p>>>(p: P, hidden_size: i64, eps: f64) -> Self {
        let p = p.borrow();
        let weight = p.var("weight", &[hidden_size], nn::Init::Const(1.0));
        MambaRMSNormGated {
            weight,
            variance_epsilon: eps,
        }
    }

    fn forward(&self, hidden_states: &Tensor, gate: Option<&Tensor>) -> Tensor {
        let mut hidden_states = hidden_states.to_kind(Kind::Float);

        if let Some(gate) = gate {
            hidden_states = &hidden_states * gate.to_kind(Kind::Float).silu();
        }

        let variance =
            hidden_states
                .pow_tensor_scalar(2)
                .mean_dim([-1].as_slice(), true, Kind::Float);
        let hidden_states = &hidden_states * (variance + self.variance_epsilon).rsqrt();

        &self.weight * &hidden_states
    }
}

/// Mamba2 Mixer layer
struct Mamba2Mixer {
    num_heads: i64,
    hidden_size: i64,
    ssm_state_size: i64,
    conv_kernel_size: i64,
    intermediate_size: i64,
    time_step_rank: i64,
    layer_idx: i64,
    use_conv_bias: bool,
    n_groups: i64,
    head_dim: i64,
    chunk_size: i64,
    time_step_limit: (f64, f64),
    time_step_min: f64,
    time_step_max: f64,
    conv_dim: i64,
    conv1d: nn::Conv1D,
    in_proj: Linear,
    dt_bias: Tensor,
    a_log: Tensor,
    norm: MambaRMSNormGated,
    d: Tensor,
    out_proj: Linear,
    activation: Activation,
}

impl Mamba2Mixer {
    fn new<'p, P: Borrow<nn::Path<'p>>>(p: P, config: &Mamba2Config, layer_idx: i64) -> Self {
        let p = p.borrow();

        let intermediate_size = config.expand * config.hidden_size;
        let time_step_rank = config.get_time_step_rank();
        let conv_dim = intermediate_size + 2 * config.n_groups * config.state_size;

        // Convolutional layer
        let conv_config = nn::ConvConfig {
            stride: 1,
            padding: config.conv_kernel - 1,
            groups: conv_dim,
            bias: config.use_conv_bias,
            ..Default::default()
        };
        let conv1d = nn::conv1d(
            p / "conv1d",
            conv_dim,
            conv_dim,
            config.conv_kernel,
            conv_config,
        );

        // Input projection
        let projection_size = intermediate_size + conv_dim + config.num_heads;
        let in_proj = nn::linear(
            p / "in_proj",
            config.hidden_size,
            projection_size,
            nn::LinearConfig {
                bias: config.use_bias,
                ..Default::default()
            },
        );

        // Time step projection bias
        let dt_bias = p.var("dt_bias", &[config.num_heads], nn::Init::Const(1.0));

        // SSM parameters
        let a_init = Tensor::arange_start(1, config.num_heads + 1, (Kind::Float, p.device()));
        let a_log = p.var_copy("A_log", &a_init.log());

        let norm = MambaRMSNormGated::new(p / "norm", intermediate_size, config.layer_norm_epsilon);
        let d = p.var("D", &[config.num_heads], nn::Init::Const(1.0));

        let out_proj = nn::linear(
            p / "out_proj",
            intermediate_size,
            config.hidden_size,
            nn::LinearConfig {
                bias: config.use_bias,
                ..Default::default()
            },
        );

        let activation = config.hidden_act;

        Mamba2Mixer {
            num_heads: config.num_heads,
            hidden_size: config.hidden_size,
            ssm_state_size: config.state_size,
            conv_kernel_size: config.conv_kernel,
            intermediate_size,
            time_step_rank,
            layer_idx,
            use_conv_bias: config.use_conv_bias,
            n_groups: config.n_groups,
            head_dim: config.head_dim,
            chunk_size: config.chunk_size,
            time_step_limit: config.time_step_limit,
            time_step_min: config.time_step_min,
            time_step_max: config.time_step_max,
            conv_dim,
            conv1d,
            in_proj,
            dt_bias,
            a_log: a_log,
            norm,
            d,
            out_proj,
            activation,
        }
    }

    fn forward_t(
        &self,
        hidden_states: &Tensor,
        mut cache: Option<&mut Mamba2Cache>,
        _cache_position: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Tensor {
        let device = hidden_states.device();
        let (batch_size, seq_len, _) = hidden_states.size3().unwrap();

        // Apply mask to padding states if provided
        let hidden_states = if let Some(mask) = attention_mask {
            if mask.size()[1] > 1 && mask.size()[0] > 1 {
                hidden_states * mask.unsqueeze(-1)
            } else {
                hidden_states.shallow_clone()
            }
        } else {
            hidden_states.shallow_clone()
        };

        // 1. Input projection
        let projected_states = hidden_states.apply(&self.in_proj);

        let groups_time_state_size = self.n_groups * self.ssm_state_size;
        let d_mlp = (projected_states.size()[2]
            - 2 * self.intermediate_size
            - 2 * groups_time_state_size
            - self.num_heads)
            / 2;

        // Split projected states
        let splits = projected_states.split_with_sizes(
            &[
                d_mlp,
                d_mlp,
                self.intermediate_size,
                self.conv_dim,
                self.num_heads,
            ],
            2,
        );
        let _x_mlp1 = &splits[0];
        let _x_mlp2 = &splits[1];
        let gate = &splits[2];
        let hidden_states_b_c = &splits[3];
        let dt = &splits[4];

        // 2. Convolution
        let hidden_states_b_c = if cache.is_some() && seq_len == 1 {
            // Single token generation with cache
            let cache_ref = cache.as_ref().unwrap();
            let conv_states = &cache_ref.conv_states[self.layer_idx as usize];
            
            // Shift conv states and append new token
            let new_conv_states = Tensor::cat(
                &[
                    &conv_states.narrow(2, 1, self.conv_kernel_size - 1),
                    &hidden_states_b_c,
                ],
                2,
            );
            
            // Update cache
            if let Some(cache_mut) = cache.as_mut() {
                cache_mut.conv_states[self.layer_idx as usize] = new_conv_states.shallow_clone();
            }
            
            // Apply convolution - use forward on the padded sequence
            let new_conv_states_t = new_conv_states.transpose(1, 2);
            let conv_out = self.conv1d.forward(&new_conv_states_t);
            // Take only the last position (new token)
            let conv_out = conv_out.select(2, -1).unsqueeze(1);
            
            // Apply activation
            match self.activation {
                Activation::swish | Activation::silu => conv_out.silu(),
                _ => self.activation.get_function().get_fn()(&conv_out),
            }
        } else {
            // Full sequence or cache initialization
            let hidden_states_b_c_t = hidden_states_b_c.transpose(1, 2);
            let conv_out = self.conv1d.forward_t(&hidden_states_b_c_t, train);
            let conv_out = conv_out.narrow(2, 0, seq_len).transpose(1, 2);
            let activated = match self.activation {
                Activation::swish | Activation::silu => conv_out.silu(),
                _ => self.activation.get_function().get_fn()(&conv_out),
            };
            
            // Initialize cache with the last conv_kernel tokens if processing prompt
            if let Some(cache_mut) = cache.as_mut() {
                if seq_len >= self.conv_kernel_size {
                    let last_tokens = hidden_states_b_c.narrow(
                        1,
                        seq_len - self.conv_kernel_size,
                        self.conv_kernel_size,
                    );
                    cache_mut.conv_states[self.layer_idx as usize] = last_tokens;
                } else {
                    // Pad with zeros if prompt is shorter than kernel
                    let padding = Tensor::zeros(
                        [batch_size, self.conv_dim, self.conv_kernel_size - seq_len],
                        (hidden_states_b_c.kind(), hidden_states_b_c.device()),
                    );
                    cache_mut.conv_states[self.layer_idx as usize] = Tensor::cat(
                        &[padding, hidden_states_b_c.shallow_clone()],
                        2,
                    );
                }
            }
            
            activated
        };

        // Apply mask after convolution
        let hidden_states_b_c = if let Some(mask) = attention_mask {
            if mask.size()[1] > 1 && mask.size()[0] > 1 {
                &hidden_states_b_c * mask.unsqueeze(-1)
            } else {
                hidden_states_b_c
            }
        } else {
            hidden_states_b_c
        };

        // Split convolution output
        let splits = hidden_states_b_c.split_with_sizes(
            &[
                self.intermediate_size,
                groups_time_state_size,
                groups_time_state_size,
            ],
            2,
        );
        let hidden_states = &splits[0];
        let b = &splits[1];
        let c = &splits[2];

        // 3. SSM transformation (simplified naive implementation)
        // This is a placeholder - full SSM scan would require custom CUDA kernels
        let a = -self.a_log.exp();

        // Time step transformation
        let dt = dt.softplus() + &self.dt_bias;
        let dt = dt.clamp(self.time_step_limit.0, self.time_step_limit.1);
        // Expand dt to include head_dim
        let dt = dt
            .unsqueeze(-1)
            .expand(&[batch_size, seq_len, self.num_heads, self.head_dim], false);

        // Reshape for SSM computation
        let hidden_states =
            hidden_states.view([batch_size, seq_len, self.num_heads, self.head_dim]);
        let b = b.view([batch_size, seq_len, self.n_groups, -1]);
        let c = c.view([batch_size, seq_len, self.n_groups, -1]);

        // Simplified SSM scan (without chunking optimization)
        // In production, this would use optimized kernels
        let mut ssm_state = Tensor::zeros(
            [
                batch_size,
                self.num_heads,
                self.head_dim,
                self.ssm_state_size,
            ],
            (Kind::Float, device),
        );

        let mut outputs = Vec::new();
        for t in 0..seq_len {
            let h_t = hidden_states.select(1, t);
            let b_t = b
                .select(1, t)
                .repeat_interleave_self_int(self.num_heads / self.n_groups, 1, None)
                .view([batch_size, self.num_heads, -1]);
            let c_t = c
                .select(1, t)
                .repeat_interleave_self_int(self.num_heads / self.n_groups, 1, None)
                .view([batch_size, self.num_heads, -1]);
            let dt_t = dt.select(1, t); // Now has shape [batch_size, num_heads, head_dim]

            // Discretize
            // a has shape [num_heads], expand to [1, num_heads, head_dim, state_size]
            let a_expanded = a.view([1, self.num_heads, 1, 1]).expand(
                &[1, self.num_heads, self.head_dim, self.ssm_state_size],
                false,
            );
            let da = (&dt_t.unsqueeze(-1) * &a_expanded).exp();

            // dt_t: [batch_size, num_heads, head_dim]
            // b_t: [batch_size, num_heads, state_size]
            // We want db: [batch_size, num_heads, head_dim, state_size]
            let db = dt_t.unsqueeze(-1) * b_t.unsqueeze(2);

            // Update state
            ssm_state = &ssm_state * &da + &h_t.unsqueeze(-1) * &db;

            // Compute output
            let y = (&ssm_state * &c_t.unsqueeze(2)).sum_dim_intlist(
                [-1].as_slice(),
                false,
                Kind::Float,
            );
            let y = &y + &h_t * &self.d.view([1, self.num_heads, 1]);
            outputs.push(y);
        }

        let scan_output = Tensor::stack(&outputs, 1);
        let scan_output = scan_output.view([batch_size, seq_len, -1]);

        // Update cache if provided
        if let Some(cache) = cache {
            cache.ssm_states[self.layer_idx as usize] = ssm_state;
        }

        // Apply normalization with gate
        let scan_output = self.norm.forward(&scan_output, Some(gate));

        // 4. Output projection
        scan_output.apply(&self.out_proj)
    }
}

/// Mamba2 Block (layer)
struct Mamba2Block {
    residual_in_fp32: bool,
    norm: MambaRMSNormGated,
    mixer: Mamba2Mixer,
}

impl Mamba2Block {
    fn new<'p, P: Borrow<nn::Path<'p>>>(p: P, config: &Mamba2Config, layer_idx: i64) -> Self {
        let p = p.borrow();

        let norm =
            MambaRMSNormGated::new(p / "norm", config.hidden_size, config.layer_norm_epsilon);
        let mixer = Mamba2Mixer::new(p / "mixer", config, layer_idx);

        Mamba2Block {
            residual_in_fp32: config.residual_in_fp32,
            norm,
            mixer,
        }
    }

    fn forward_t(
        &self,
        hidden_states: &Tensor,
        mut cache: Option<&mut Mamba2Cache>,
        _cache_position: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Tensor {
        let residual = if self.residual_in_fp32 {
            hidden_states.to_kind(Kind::Float)
        } else {
            hidden_states.shallow_clone()
        };

        let hidden_states = self.norm.forward(hidden_states, None);
        let hidden_states =
            self.mixer
                .forward_t(&hidden_states, cache, _cache_position, attention_mask, train);

        residual + hidden_states
    }
}

/// Base Mamba2 model
pub struct Mamba2Model {
    embeddings: nn::Embedding,
    layers: Vec<Mamba2Block>,
    norm_f: MambaRMSNormGated,
    gradient_checkpointing: bool,
}

impl Mamba2Model {
    pub fn new<'p, P: Borrow<nn::Path<'p>>>(p: P, config: &Mamba2Config) -> Self {
        let p = p.borrow();

        let embeddings = embedding(
            p / "embeddings",
            config.vocab_size,
            config.hidden_size,
            Default::default(),
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers {
            layers.push(Mamba2Block::new(p / "layers" / i, config, i));
        }

        let norm_f =
            MambaRMSNormGated::new(p / "norm_f", config.hidden_size, config.layer_norm_epsilon);

        Mamba2Model {
            embeddings,
            layers,
            norm_f,
            gradient_checkpointing: false,
        }
    }

    pub fn forward_t<'a>(
        &self,
        input_ids: &Tensor,
        mut cache: Option<&'a mut Mamba2Cache>,
        _cache_position: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Mamba2ModelOutput<'a> {
        let mut hidden_states = input_ids.apply(&self.embeddings);

        let mut all_hidden_states = Vec::new();

        for layer in self.layers.iter() {
            all_hidden_states.push(hidden_states.shallow_clone());

            hidden_states = layer.forward_t(
                &hidden_states,
                cache.as_deref_mut(),
                _cache_position,
                attention_mask,
                train,
            );
        }

        let hidden_states = self.norm_f.forward(&hidden_states, None);
        all_hidden_states.push(hidden_states.shallow_clone());

        Mamba2ModelOutput {
            hidden_states,
            all_hidden_states: Some(all_hidden_states),
            cache,
        }
    }
}

#[derive(Debug)]
pub struct Mamba2ModelOutput<'a> {
    pub hidden_states: Tensor,
    pub all_hidden_states: Option<Vec<Tensor>>,
    pub cache: Option<&'a mut Mamba2Cache>,
}

/// Mamba2 for Causal Language Modeling
pub struct Mamba2ForCausalLM {
    model: Mamba2Model,
    lm_head: Linear,
    tie_word_embeddings: bool,
}

impl Mamba2ForCausalLM {
    pub fn new<'p, P: Borrow<nn::Path<'p>>>(p: P, config: &Mamba2Config) -> Self {
        let p = p.borrow();

        let model = Mamba2Model::new(p / "backbone", config);
        let lm_head = nn::linear(
            p / "lm_head",
            config.hidden_size,
            config.vocab_size,
            nn::LinearConfig {
                bias: config.use_bias,
                ..Default::default()
            },
        );

        Mamba2ForCausalLM {
            model,
            lm_head,
            tie_word_embeddings: config.tie_word_embeddings,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mut cache: Option<&mut Mamba2Cache>,
        _cache_position: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<LMModelOutput, RustBertError> {
        let input_ids = input_ids.ok_or(RustBertError::ValueError(
            "input_ids must be provided".into(),
        ))?;

        let outputs = self
            .model
            .forward_t(input_ids, cache, _cache_position, attention_mask, train);

        let logits = if self.tie_word_embeddings {
            outputs.hidden_states.matmul(&self.model.embeddings.ws.tr())
        } else {
            outputs.hidden_states.apply(&self.lm_head)
        };

        Ok(LMModelOutput {
            lm_logits: logits,
            cache: Cache::None,
        })
    }
}

// Language generation support will be added in a future update
