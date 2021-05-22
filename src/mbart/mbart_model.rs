use crate::{Activation, Config};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// # MBART Pretrained model weight files
pub struct MBartModelResources;

/// # MBART Pretrained model config files
pub struct MBartConfigResources;

/// # MBART Pretrained model vocab files
pub struct MBartVocabResources;

impl MBartModelResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const MBART50_MANY_TO_MANY: (&'static str, &'static str) = (
        "mbart/model",
        "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/rust_model.ot",
    );
}

impl MBartConfigResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const MBART50_MANY_TO_MANY: (&'static str, &'static str) = (
        "mbart/config",
        "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/config.json",
    );
}

impl MBartVocabResources {
    /// Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    pub const MBART50_MANY_TO_MANY: (&'static str, &'static str) = (
        "mbart/vocab",
        "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/resolve/main/sentencepiece.bpe.model",
    );
}

#[derive(Debug, Serialize, Deserialize, Clone)]
/// # MBART model configuration
/// Defines the MBART model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct MBartConfig {
    pub vocab_size: i64,
    pub max_position_embeddings: i64,
    pub encoder_layers: i64,
    pub encoder_attention_heads: i64,
    pub encoder_ffn_dim: i64,
    pub encoder_layerdrop: f64,
    pub decoder_layers: i64,
    pub decoder_ffn_dim: i64,
    pub decoder_attention_heads: i64,
    pub decoder_layerdrop: f64,
    pub is_encoder_decoder: Option<bool>,
    pub activation_function: Option<Activation>,
    pub d_model: i64,
    pub dropout: f64,
    pub activation_dropout: f64,
    pub attention_dropout: f64,
    pub classifier_dropout: Option<f64>,
    pub scale_embedding: Option<bool>,
    pub bos_token_id: Option<i64>,
    pub eos_token_id: Option<i64>,
    pub pad_token_id: Option<i64>,
    pub forced_eos_token_id: Option<i64>,
    pub decoder_start_token_id: Option<i64>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub init_std: f64,
    pub min_length: Option<i64>,
    pub no_repeat_ngram_size: Option<i64>,
    pub normalize_embedding: Option<bool>,
    pub num_hidden_layers: i64,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub output_past: Option<bool>,
    pub static_position_embeddings: Option<bool>,
}

impl Config<MBartConfig> for MBartConfig {}
