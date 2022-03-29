use std::path::PathBuf;

use rust_tokenizers::tokenizer::TruncationStrategy;
use tch::{nn, Device, Tensor};

use crate::sbert::config::{
    SBertModelConfig, SBertModule, SBertModuleType, SBertModulesConfig, SBertTokenizerConfig,
};
use crate::sbert::layers::{Dense, Pooling};
use crate::sbert::transformer::SBertTransformer;
use crate::{Config, RustBertError};

pub type Attention = Vec<f32>; // Length = sequence length
pub type AttentionHead = Vec<Attention>; // Length = sequence length
pub type AttentionLayer = Vec<AttentionHead>; // Length = number of heads per attention layer
pub type AttentionOutput = Vec<AttentionLayer>; // Length = number of attention layers

pub type Embedding = Vec<f32>;

pub struct SBertModel<T> {
    conf_tokenizer: SBertTokenizerConfig,
    truncation_strategy: TruncationStrategy,
    conf_model: SBertModelConfig,
    var_store: nn::VarStore,
    transformer: T,
    pooling_layer: Pooling,
    dense_layer: Option<Dense>,
    normalize_embeddings: bool,
}

impl<T: SBertTransformer> SBertModel<T> {
    pub fn new<P: Into<PathBuf>>(model_dir: P, device: Device) -> Result<Self, RustBertError> {
        let model_dir = model_dir.into();

        let conf_model = T::model_conf(&model_dir);
        let conf_tokenizer = T::tokenizer_conf(&model_dir);
        let modules = SBertModulesConfig::from_file(model_dir.join("modules.json"));

        let (transformer, var_store) = match modules.get(0) {
            Some(SBertModule {
                mod_type: SBertModuleType::Transformer,
                ..
            }) => T::from(&model_dir, device)?,
            Some(_) => {
                return Err(RustBertError::InvalidConfigurationError(
                    "First module defined in modules.json must be a Transformer".to_string(),
                ));
            }
            None => {
                return Err(RustBertError::InvalidConfigurationError(
                    "No modules found in modules.json".to_string(),
                ));
            }
        };

        let pooling_layer = match modules.get(1) {
            Some(SBertModule {
                mod_type: SBertModuleType::Pooling,
                path,
                ..
            }) => Pooling::new(model_dir.join(&path)),
            Some(_) => {
                return Err(RustBertError::InvalidConfigurationError(
                    "Second module defined in modules.json must be a Pooling".to_string(),
                ));
            }
            None => {
                return Err(RustBertError::InvalidConfigurationError(
                    "Pooling module not found in second position in modules.json".to_string(),
                ));
            }
        };

        let mut dense_layer = None;
        let mut normalize_embeddings = false;
        for i in 2..=3 {
            match modules.get(i) {
                Some(SBertModule {
                    mod_type: SBertModuleType::Dense,
                    path,
                    ..
                }) => {
                    dense_layer = Some(Dense::new(model_dir.join(&path), device)?);
                }
                Some(SBertModule {
                    mod_type: SBertModuleType::Normalize,
                    ..
                }) => {
                    normalize_embeddings = true;
                }
                _ => (),
            }
        }

        Ok(SBertModel {
            conf_tokenizer,
            truncation_strategy: TruncationStrategy::LongestFirst,
            conf_model,
            var_store,
            transformer,
            pooling_layer,
            dense_layer,
            normalize_embeddings,
        })
    }

    pub fn set_tokenizer_truncation(&mut self, truncation_strategy: TruncationStrategy) {
        self.truncation_strategy = truncation_strategy;
    }

    pub fn tokenize<S: AsRef<str>>(&self, inputs: &[S]) -> SBertTokenizerOuput {
        let tokenized_input = self.transformer.tokenize(
            inputs,
            self.conf_tokenizer.max_seq_length,
            &self.truncation_strategy,
            0,
        );

        let max_len = tokenized_input
            .iter()
            .map(|input| input.token_ids.len())
            .max()
            .unwrap_or(0);

        let pad_token_id = self.conf_model.pad_token_id();
        let tokens_ids = tokenized_input
            .into_iter()
            .map(|input| {
                let mut token_ids = input.token_ids;
                token_ids.extend(vec![pad_token_id; max_len - token_ids.len()]);
                token_ids
            })
            .collect::<Vec<_>>();

        let tokens_masks = tokens_ids
            .iter()
            .map(|input| {
                Tensor::of_slice(
                    &input
                        .iter()
                        .map(|&e| {
                            if e == pad_token_id {
                                0 as i64
                            } else {
                                1 as i64
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        let tokens_ids = tokens_ids
            .into_iter()
            .map(|input| Tensor::of_slice(&(input)))
            .collect::<Vec<_>>();

        SBertTokenizerOuput {
            tokens_ids,
            tokens_masks,
        }
    }

    pub fn forward<S: AsRef<str>>(&self, inputs: &[S]) -> Result<SBertModelOuput, RustBertError> {
        let SBertTokenizerOuput {
            tokens_ids,
            tokens_masks,
        } = self.tokenize(&inputs);
        let tokens_ids = Tensor::stack(&tokens_ids, 0).to(self.var_store.device());
        let tokens_masks = Tensor::stack(&tokens_masks, 0).to(self.var_store.device());

        let (tokens_embeddings, all_attentions) =
            self.transformer.forward(&tokens_ids, &tokens_masks)?;

        let mean_pool = self.pooling_layer.forward(tokens_embeddings, &tokens_masks);
        let maybe_linear = if let Some(dense_layer) = &self.dense_layer {
            dense_layer.forward(&mean_pool)
        } else {
            mean_pool
        };
        let maybe_normalized = if self.normalize_embeddings {
            let norm = &maybe_linear
                .norm_scalaropt_dim(2, &[1], true)
                .clamp_min(1e-12)
                .expand_as(&maybe_linear);
            maybe_linear / norm
        } else {
            maybe_linear
        };

        Ok(SBertModelOuput {
            embeddings: maybe_normalized,
            all_attentions,
        })
    }

    pub fn encode<S>(&self, inputs: &[S]) -> Result<Vec<Embedding>, RustBertError>
    where
        S: AsRef<str>,
    {
        let SBertModelOuput { embeddings, .. } = self.forward(inputs)?;
        Ok(Vec::from(embeddings))
    }

    pub fn encode_with_attention<S>(
        &self,
        inputs: &[S],
    ) -> Result<(Vec<Embedding>, Vec<AttentionOutput>), RustBertError>
    where
        S: AsRef<str>,
    {
        let SBertModelOuput {
            embeddings,
            all_attentions,
        } = self.forward(inputs)?;

        let embeddings = Vec::from(embeddings);
        let all_attentions = all_attentions.ok_or(RustBertError::InvalidConfigurationError(
            "No attention outputted".into(),
        ))?;

        let attention_outputs = (0..inputs.len() as i64)
            .map(|i| {
                let mut attention_output =
                    AttentionOutput::with_capacity(self.conf_model.nb_layers());
                for layer in all_attentions.iter() {
                    let mut attention_layer =
                        AttentionLayer::with_capacity(self.conf_model.nb_heads());
                    for head in 0..self.conf_model.nb_heads() {
                        let attention_slice = layer
                            .slice(0, i, i + 1, 1)
                            .slice(1, head as i64, head as i64 + 1, 1)
                            .squeeze();
                        let attention_head = AttentionHead::from(attention_slice);
                        attention_layer.push(attention_head);
                    }
                    attention_output.push(attention_layer);
                }
                attention_output
            })
            .collect::<Vec<AttentionOutput>>();

        Ok((embeddings, attention_outputs))
    }
}

pub struct SBertTokenizerOuput {
    pub tokens_ids: Vec<Tensor>,
    pub tokens_masks: Vec<Tensor>,
}

pub struct SBertModelOuput {
    pub embeddings: Tensor,
    pub all_attentions: Option<Vec<Tensor>>,
}
