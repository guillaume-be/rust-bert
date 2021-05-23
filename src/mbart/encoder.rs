use crate::common::activations::TensorFunction;
use crate::common::dropout::Dropout;
use crate::mbart::attention::MBartAttention;
use crate::mbart::MBartConfig;
use crate::Activation;
use std::borrow::Borrow;
use tch::{nn, Tensor};

pub struct MBartEncoderLayer {
    self_attention: MBartAttention,
    self_attention_layer_norm: nn::LayerNorm,
    dropout: Dropout,
    activation_dropout: Dropout,
    activation: TensorFunction,
    fc1: nn::Linear,
    fc2: nn::Linear,
    final_layer_norm: nn::LayerNorm,
}

impl MBartEncoderLayer {
    pub fn new<'p, P>(p: P, config: &MBartConfig) -> MBartEncoderLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let layer_norm_config = nn::LayerNormConfig {
            eps: 1e-5,
            ..Default::default()
        };
        let output_attention = config.output_attentions.unwrap_or(false);
        let self_attention = MBartAttention::new(
            p / "self_attn",
            config.d_model,
            config.encoder_attention_heads,
            config.attention_dropout,
            false,
            false,
            output_attention,
        );
        let self_attention_layer_norm = nn::layer_norm(
            p / "self_attn_layer_norm",
            vec![config.d_model],
            layer_norm_config,
        );
        let dropout = Dropout::new(config.dropout);
        let activation_dropout = Dropout::new(config.activation_dropout);
        let activation_function = match &config.activation_function {
            Some(act_function) => act_function,
            None => &Activation::gelu,
        };
        let activation = activation_function.get_function();
        let fc1 = nn::linear(
            p / "fc1",
            config.d_model,
            config.encoder_ffn_dim,
            Default::default(),
        );
        let fc2 = nn::linear(
            p / "fc2",
            config.encoder_ffn_dim,
            config.d_model,
            Default::default(),
        );

        let final_layer_norm = nn::layer_norm(
            p / "final_layer_norm",
            vec![config.d_model],
            layer_norm_config,
        );

        MBartEncoderLayer {
            self_attention,
            self_attention_layer_norm,
            dropout,
            activation_dropout,
            activation,
            fc1,
            fc2,
            final_layer_norm,
        }
    }

    pub fn forward_t(
        &self,
        x: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        train: bool,
    ) -> (Tensor, Option<Tensor>) {
        let output = x.apply(&self.self_attention_layer_norm);
        let (output, attention_weights, _) =
            self.self_attention
                .forward_t(&output, None, encoder_attention_mask, None, train);
        let output: Tensor = output.apply_t(&self.dropout, train) + x;

        let residual = output.copy();
        let output = output.apply(&self.final_layer_norm);
        let output = (self.activation.get_fn())(&output.apply(&self.fc1));
        let output = output
            .apply_t(&self.activation_dropout, train)
            .apply(&self.fc2)
            .apply_t(&self.dropout, train);
        let output = output + residual;
        (output, attention_weights)
    }
}
