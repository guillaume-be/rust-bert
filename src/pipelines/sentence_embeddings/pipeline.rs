use std::borrow::Borrow;
use std::convert::{TryFrom, TryInto};

use rust_tokenizers::tokenizer::TruncationStrategy;
use tch::{nn, Tensor};

use crate::albert::AlbertForSentenceEmbeddings;
use crate::bert::BertForSentenceEmbeddings;
use crate::distilbert::DistilBertForSentenceEmbeddings;
use crate::pipelines::common::{ConfigOption, ModelType, TokenizerOption};
use crate::pipelines::sentence_embeddings::layers::{Dense, DenseConfig, Pooling, PoolingConfig};
use crate::pipelines::sentence_embeddings::{
    AttentionHead, AttentionLayer, AttentionOutput, Embedding, SentenceEmbeddingsConfig,
    SentenceEmbeddingsModulesConfig, SentenceEmbeddingsSentenceBertConfig,
    SentenceEmbeddingsTokenizerConfig,
};
use crate::roberta::RobertaForSentenceEmbeddings;
use crate::t5::T5ForSentenceEmbeddings;
use crate::{Config, RustBertError};

/// # Abstraction that holds one particular sentence embeddings model, for any of the supported models
pub enum SentenceEmbeddingsOption {
    /// Bert for Sentence Embeddings
    Bert(BertForSentenceEmbeddings),
    /// DistilBert for Sentence Embeddings
    DistilBert(DistilBertForSentenceEmbeddings),
    /// Roberta for Sentence Embeddings
    Roberta(RobertaForSentenceEmbeddings),
    /// Albert for Sentence Embeddings
    Albert(AlbertForSentenceEmbeddings),
    /// T5 for Sentence Embeddings
    T5(T5ForSentenceEmbeddings),
}

impl SentenceEmbeddingsOption {
    /// Instantiate a new sentence embeddings transformer of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `transformer_type` - `ModelType` indicating the transformer model type to load (must match with the actual data to be loaded)
    /// * `p` - `tch::nn::Path` path to the model file to load (e.g. rust_model.ot)
    /// * `config` - A configuration (the transformer model type of the configuration must be compatible with the value for `transformer_type`)
    pub fn new<'p, P>(
        transformer_type: ModelType,
        p: P,
        config: &ConfigOption,
    ) -> Result<Self, RustBertError>
    where
        P: Borrow<nn::Path<'p>>,
    {
        use SentenceEmbeddingsOption::*;

        let option = match transformer_type {
            ModelType::Bert => Bert(BertForSentenceEmbeddings::new(p, &(config.try_into()?))),
            ModelType::DistilBert => DistilBert(DistilBertForSentenceEmbeddings::new(
                p,
                &(config.try_into()?),
            )),
            ModelType::Roberta => Roberta(RobertaForSentenceEmbeddings::new_with_optional_pooler(
                p,
                &(config.try_into()?),
                false,
            )),
            ModelType::Albert => Albert(AlbertForSentenceEmbeddings::new(p, &(config.try_into()?))),
            ModelType::T5 => T5(T5ForSentenceEmbeddings::new(p, &(config.try_into()?))),
            _ => {
                return Err(RustBertError::InvalidConfigurationError(format!(
                    "Unsupported transformer model {transformer_type:?} for Sentence Embeddings"
                )));
            }
        };

        Ok(option)
    }

    /// Interface method to forward() of the particular transformer models.
    pub fn forward(
        &self,
        tokens_ids: &Tensor,
        tokens_masks: &Tensor,
    ) -> Result<(Tensor, Option<Vec<Tensor>>), RustBertError> {
        match self {
            Self::Bert(transformer) => transformer
                .forward_t(
                    Some(tokens_ids),
                    Some(tokens_masks),
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                )
                .map(|transformer_output| {
                    (
                        transformer_output.hidden_state,
                        transformer_output.all_attentions,
                    )
                }),
            Self::DistilBert(transformer) => transformer
                .forward_t(Some(tokens_ids), Some(tokens_masks), None, false)
                .map(|transformer_output| {
                    (
                        transformer_output.hidden_state,
                        transformer_output.all_attentions,
                    )
                }),
            Self::Roberta(transformer) => transformer
                .forward_t(
                    Some(tokens_ids),
                    Some(tokens_masks),
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                )
                .map(|transformer_output| {
                    (
                        transformer_output.hidden_state,
                        transformer_output.all_attentions,
                    )
                }),
            Self::Albert(transformer) => transformer
                .forward_t(
                    Some(tokens_ids),
                    Some(tokens_masks),
                    None,
                    None,
                    None,
                    false,
                )
                .map(|transformer_output| {
                    (
                        transformer_output.hidden_state,
                        transformer_output.all_attentions.map(|attentions| {
                            attentions
                                .into_iter()
                                .map(|tensors| {
                                    let num_inner_groups = tensors.len() as f64;
                                    tensors.into_iter().sum::<Tensor>() / num_inner_groups
                                })
                                .collect()
                        }),
                    )
                }),
            Self::T5(transformer) => transformer.forward(tokens_ids, tokens_masks),
        }
    }
}

/// # SentenceEmbeddingsModel to perform sentence embeddings
///
/// It is made of the following blocks:
/// - `transformer`: Base transformer model
/// - `pooling`: Pooling layer
/// - `dense` _(optional)_: Linear (feed forward) layer
/// - `normalization` _(optional)_: Embeddings normalization
pub struct SentenceEmbeddingsModel {
    sentence_bert_config: SentenceEmbeddingsSentenceBertConfig,
    tokenizer: TokenizerOption,
    tokenizer_truncation_strategy: TruncationStrategy,
    var_store: nn::VarStore,
    transformer: SentenceEmbeddingsOption,
    transformer_config: ConfigOption,
    pooling_layer: Pooling,
    dense_layer: Option<Dense>,
    normalize_embeddings: bool,
    embeddings_dim: i64,
}

impl SentenceEmbeddingsModel {
    /// Build a new `SentenceEmbeddingsModel`
    ///
    /// # Arguments
    ///
    /// * `config` - `SentenceEmbeddingsConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    pub fn new(config: SentenceEmbeddingsConfig) -> Result<Self, RustBertError> {
        let SentenceEmbeddingsConfig {
            modules_config_resource,
            sentence_bert_config_resource,
            tokenizer_config_resource,
            tokenizer_vocab_resource,
            tokenizer_merges_resource,
            transformer_type,
            transformer_config_resource,
            transformer_weights_resource,
            pooling_config_resource,
            dense_config_resource,
            dense_weights_resource,
            device,
        } = config;

        let modules =
            SentenceEmbeddingsModulesConfig::from_file(modules_config_resource.get_local_path()?)
                .validate()?;

        // Setup tokenizer
        let tokenizer_config = SentenceEmbeddingsTokenizerConfig::from_file(
            tokenizer_config_resource.get_local_path()?,
        );
        let sentence_bert_config = SentenceEmbeddingsSentenceBertConfig::from_file(
            sentence_bert_config_resource.get_local_path()?,
        );
        let tokenizer = TokenizerOption::from_file(
            transformer_type,
            tokenizer_vocab_resource
                .get_local_path()?
                .to_string_lossy()
                .as_ref(),
            tokenizer_merges_resource
                .as_ref()
                .map(|resource| resource.get_local_path())
                .transpose()?
                .map(|path| path.to_string_lossy().into_owned())
                .as_deref(),
            tokenizer_config
                .do_lower_case
                .unwrap_or(sentence_bert_config.do_lower_case),
            tokenizer_config.strip_accents,
            tokenizer_config.add_prefix_space,
        )?;

        // Setup transformer
        let mut var_store = nn::VarStore::new(device);
        let transformer_config = ConfigOption::from_file(
            transformer_type,
            transformer_config_resource.get_local_path()?,
        );
        let transformer =
            SentenceEmbeddingsOption::new(transformer_type, var_store.root(), &transformer_config)?;
        var_store.load(transformer_weights_resource.get_local_path()?)?;

        // Setup pooling layer
        let pooling_config = PoolingConfig::from_file(pooling_config_resource.get_local_path()?);
        let mut embeddings_dim = pooling_config.word_embedding_dimension;
        let pooling_layer = Pooling::new(pooling_config);

        // Setup dense layer
        let dense_layer = if modules.dense_module().is_some() {
            let dense_config =
                DenseConfig::from_file(dense_config_resource.unwrap().get_local_path()?);
            embeddings_dim = dense_config.out_features;
            Some(Dense::new(
                dense_config,
                dense_weights_resource.unwrap().get_local_path()?,
                device,
            )?)
        } else {
            None
        };

        let normalize_embeddings = modules.has_normalization();

        Ok(Self {
            tokenizer,
            sentence_bert_config,
            tokenizer_truncation_strategy: TruncationStrategy::LongestFirst,
            var_store,
            transformer,
            transformer_config,
            pooling_layer,
            dense_layer,
            normalize_embeddings,
            embeddings_dim,
        })
    }

    /// Get a reference to the model tokenizer.
    pub fn get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
    }

    /// Get a mutable reference to the model tokenizer.
    pub fn get_tokenizer_mut(&mut self) -> &mut TokenizerOption {
        &mut self.tokenizer
    }

    /// Sets the tokenizer's truncation strategy
    pub fn set_tokenizer_truncation(&mut self, truncation_strategy: TruncationStrategy) {
        self.tokenizer_truncation_strategy = truncation_strategy;
    }

    /// Return the embedding output dimension
    pub fn get_embedding_dim(&self) -> Result<i64, RustBertError> {
        Ok(self.embeddings_dim)
    }

    /// Tokenizes the inputs
    pub fn tokenize<S>(&self, inputs: &[S]) -> SentenceEmbeddingsTokenizerOutput
    where
        S: AsRef<str> + Sync,
    {
        let tokenized_input = self.tokenizer.encode_list(
            inputs,
            self.sentence_bert_config.max_seq_length,
            &self.tokenizer_truncation_strategy,
            0,
        );

        let max_len = tokenized_input
            .iter()
            .map(|input| input.token_ids.len())
            .max()
            .unwrap_or(0);

        let pad_token_id = self.tokenizer.get_pad_id().unwrap_or(0);
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
                Tensor::from_slice(
                    &input
                        .iter()
                        .map(|&e| i64::from(e != pad_token_id))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();

        let tokens_ids = tokens_ids
            .into_iter()
            .map(|input| Tensor::from_slice(&(input)))
            .collect::<Vec<_>>();

        SentenceEmbeddingsTokenizerOutput {
            tokens_ids,
            tokens_masks,
        }
    }

    /// Computes sentence embeddings, outputs `Tensor`.
    pub fn encode_as_tensor<S>(
        &self,
        inputs: &[S],
    ) -> Result<SentenceEmbeddingsModelOutput, RustBertError>
    where
        S: AsRef<str> + Sync,
    {
        let SentenceEmbeddingsTokenizerOutput {
            tokens_ids,
            tokens_masks,
        } = self.tokenize(inputs);
        let tokens_ids = Tensor::stack(&tokens_ids, 0).to(self.var_store.device());
        let tokens_masks = Tensor::stack(&tokens_masks, 0).to(self.var_store.device());

        let (tokens_embeddings, all_attentions) =
            tch::no_grad(|| self.transformer.forward(&tokens_ids, &tokens_masks))?;

        let mean_pool =
            tch::no_grad(|| self.pooling_layer.forward(tokens_embeddings, &tokens_masks));
        let maybe_linear = if let Some(dense_layer) = &self.dense_layer {
            tch::no_grad(|| dense_layer.forward(&mean_pool))
        } else {
            mean_pool
        };
        let maybe_normalized = if self.normalize_embeddings {
            let norm = &maybe_linear
                .norm_scalaropt_dim(2, [1], true)
                .clamp_min(1e-12)
                .expand_as(&maybe_linear);
            maybe_linear / norm
        } else {
            maybe_linear
        };

        Ok(SentenceEmbeddingsModelOutput {
            embeddings: maybe_normalized,
            all_attentions,
        })
    }

    /// Computes sentence embeddings.
    pub fn encode<S>(&self, inputs: &[S]) -> Result<Vec<Embedding>, RustBertError>
    where
        S: AsRef<str> + Sync,
    {
        let SentenceEmbeddingsModelOutput { embeddings, .. } = self.encode_as_tensor(inputs)?;
        Ok(Vec::try_from(embeddings)?)
    }

    fn nb_layers(&self) -> usize {
        use SentenceEmbeddingsOption::*;
        match (&self.transformer, &self.transformer_config) {
            (Bert(_), ConfigOption::Bert(conf)) => conf.num_hidden_layers as usize,
            (Bert(_), _) => unreachable!(),
            (DistilBert(_), ConfigOption::DistilBert(conf)) => conf.n_layers as usize,
            (DistilBert(_), _) => unreachable!(),
            (Roberta(_), ConfigOption::Bert(conf)) => conf.num_hidden_layers as usize,
            (Roberta(_), _) => unreachable!(),
            (Albert(_), ConfigOption::Albert(conf)) => conf.num_hidden_layers as usize,
            (Albert(_), _) => unreachable!(),
            (T5(_), ConfigOption::T5(conf)) => conf.num_layers as usize,
            (T5(_), _) => unreachable!(),
        }
    }

    fn nb_heads(&self) -> usize {
        use SentenceEmbeddingsOption::*;
        match (&self.transformer, &self.transformer_config) {
            (Bert(_), ConfigOption::Bert(conf)) => conf.num_attention_heads as usize,
            (Bert(_), _) => unreachable!(),
            (DistilBert(_), ConfigOption::DistilBert(conf)) => conf.n_heads as usize,
            (DistilBert(_), _) => unreachable!(),
            (Roberta(_), ConfigOption::Roberta(conf)) => conf.num_attention_heads as usize,
            (Roberta(_), _) => unreachable!(),
            (Albert(_), ConfigOption::Albert(conf)) => conf.num_attention_heads as usize,
            (Albert(_), _) => unreachable!(),
            (T5(_), ConfigOption::T5(conf)) => conf.num_heads as usize,
            (T5(_), _) => unreachable!(),
        }
    }

    /// Computes sentence embeddings, also outputs `AttentionOutput`s.
    pub fn encode_with_attention<S>(
        &self,
        inputs: &[S],
    ) -> Result<(Vec<Embedding>, Vec<AttentionOutput>), RustBertError>
    where
        S: AsRef<str> + Sync,
    {
        let SentenceEmbeddingsModelOutput {
            embeddings,
            all_attentions,
        } = self.encode_as_tensor(inputs)?;

        let embeddings = Vec::try_from(embeddings)?;
        let all_attentions = all_attentions.ok_or_else(|| {
            RustBertError::InvalidConfigurationError("No attention outputted".into())
        })?;

        let attention_outputs = (0..inputs.len() as i64)
            .map(|i| {
                let mut attention_output = AttentionOutput::with_capacity(self.nb_layers());
                for layer in all_attentions.iter() {
                    let mut attention_layer = AttentionLayer::with_capacity(self.nb_heads());
                    for head in 0..self.nb_heads() {
                        let attention_slice = layer
                            .slice(0, i, i + 1, 1)
                            .slice(1, head as i64, head as i64 + 1, 1)
                            .squeeze();
                        let attention_head = AttentionHead::try_from(attention_slice).unwrap();
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

/// Container for the SentenceEmbeddings tokenizer output.
pub struct SentenceEmbeddingsTokenizerOutput {
    pub tokens_ids: Vec<Tensor>,
    pub tokens_masks: Vec<Tensor>,
}

/// Container for the SentenceEmbeddings model output.
pub struct SentenceEmbeddingsModelOutput {
    pub embeddings: Tensor,
    pub all_attentions: Option<Vec<Tensor>>,
}
