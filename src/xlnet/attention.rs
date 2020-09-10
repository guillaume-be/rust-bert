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

use crate::common::dropout::Dropout;
use crate::xlnet::XLNetConfig;
use std::borrow::Borrow;
use tch::nn::Init;
use tch::{nn, Kind, Tensor};

#[derive(Debug)]
pub struct XLNetRelativeAttention {
    num_attention_heads: i64,
    attention_head_size: i64,
    hidden_size: i64,
    dropout: Dropout,
    output_attentions: bool,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    r: Tensor,
    r_r_bias: Tensor,
    r_s_bias: Tensor,
    r_w_bias: Tensor,
    seg_embed: Tensor,
    layer_norm: nn::LayerNorm,
}

impl XLNetRelativeAttention {
    pub fn new<'p, P>(p: P, config: &XLNetConfig) -> XLNetRelativeAttention
    where
        P: Borrow<nn::Path<'p>>,
    {
        assert_eq!(
            config.d_model % config.d_head,
            0,
            "Hidden size not a multiple of attention heads dimension"
        );
        let p = p.borrow();

        let q = p.var(
            "q",
            &[config.d_model, config.n_head, config.d_head],
            Init::KaimingUniform,
        );

        let k = p.var(
            "k",
            &[config.d_model, config.n_head, config.d_head],
            Init::KaimingUniform,
        );

        let v = p.var(
            "v",
            &[config.d_model, config.n_head, config.d_head],
            Init::KaimingUniform,
        );

        let o = p.var(
            "o",
            &[config.d_model, config.n_head, config.d_head],
            Init::KaimingUniform,
        );

        let r = p.var(
            "r",
            &[config.d_model, config.n_head, config.d_head],
            Init::KaimingUniform,
        );

        let r_r_bias = p.var(
            "r_r_bias",
            &[config.n_head, config.d_head],
            Init::KaimingUniform,
        );
        let r_s_bias = p.var(
            "r_s_bias",
            &[config.n_head, config.d_head],
            Init::KaimingUniform,
        );
        let r_w_bias = p.var(
            "r_w_bias",
            &[config.n_head, config.d_head],
            Init::KaimingUniform,
        );
        let seg_embed = p.var(
            "seg_embed",
            &[config.n_head, config.d_head],
            Init::KaimingUniform,
        );

        let dropout = Dropout::new(config.dropout);
        let output_attentions = match config.output_attentions {
            Some(value) => value,
            None => false,
        };
        let layer_norm_eps = match config.layer_norm_eps {
            Some(value) => value,
            None => 1e-12,
        };
        let layer_norm_config = nn::LayerNormConfig {
            eps: layer_norm_eps,
            ..Default::default()
        };
        let layer_norm = nn::layer_norm(p / "layer_norm", vec![config.d_model], layer_norm_config);

        XLNetRelativeAttention {
            num_attention_heads: config.n_head,
            attention_head_size: config.d_head,
            hidden_size: config.d_model,
            dropout,
            output_attentions,
            q,
            k,
            v,
            o,
            r,
            r_r_bias,
            r_s_bias,
            r_w_bias,
            seg_embed,
            layer_norm,
        }
    }

    fn rel_shift(&self, x: &Tensor, klen: i64) -> Tensor {
        let shape = x.size();
        x.reshape(&[shape[1], shape[0], shape[2], shape[3]])
            .narrow(0, 1, shape[1] - 1)
            .reshape(&[shape[0], shape[1] - 1, shape[2], shape[3]])
            .index_select(1, &Tensor::arange(klen, (Kind::Int64, x.device())))
    }

    fn rel_shift_bnij(&self, x: &Tensor, klen: i64) -> Tensor {
        let shape = x.size();
        x.reshape(&[shape[0], shape[1], shape[3], shape[2]])
            .narrow(2, 1, shape[1] - 1)
            .reshape(&[shape[0], shape[1], shape[2], shape[3] - 1])
            .index_select(1, &Tensor::arange(klen, (Kind::Int64, x.device())))
    }

    fn rel_attention_core(
        &self,
        q_head: &Tensor,
        k_head_h: &Tensor,
        v_head_h: &Tensor,
        k_head_r: &Tensor,
        seg_mat: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> (Tensor, Option<Tensor>) {
        let ac = Tensor::einsum("ibnd,jbnd->bnij", &[&(q_head + &self.r_w_bias), &k_head_h]);
        let bd = self.rel_shift_bnij(
            &Tensor::einsum("ibnd,jbnd->bnij", &[&(q_head + &self.r_r_bias), &k_head_r]),
            ac.size()[3],
        );

        let ef = match seg_mat {
            Some(seg_mat) => {
                let ef = Tensor::einsum(
                    "ibnd,snd->ibns",
                    &[&(q_head + &self.r_s_bias), &self.seg_embed],
                );
                Tensor::einsum("ijbs,ibns->bnij", &[seg_mat, &ef])
            }
            None => Tensor::zeros(&[1], (Kind::Float, ac.device())),
        };
    }
}
