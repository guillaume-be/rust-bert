use serde::{Deserialize, Serialize};
use tch::Device;

use crate::pipelines::common::ModelType;
use crate::resources::ResourceProvider;
use crate::{Config, RustBertError};

#[cfg(feature = "remote")]
use crate::{
    albert::{AlbertConfigResources, AlbertModelResources, AlbertVocabResources},
    bert::{BertConfigResources, BertModelResources, BertVocabResources},
    distilbert::{DistilBertConfigResources, DistilBertModelResources, DistilBertVocabResources},
    pipelines::sentence_embeddings::resources::{
        SentenceEmbeddingsConfigResources, SentenceEmbeddingsModelType,
        SentenceEmbeddingsModulesConfigResources, SentenceEmbeddingsPoolingConfigResources,
        SentenceEmbeddingsTokenizerConfigResources,
    },
    pipelines::sentence_embeddings::{
        SentenceEmbeddingsDenseConfigResources, SentenceEmbeddingsDenseResources,
    },
    resources::RemoteResource,
    roberta::{
        RobertaConfigResources, RobertaMergesResources, RobertaModelResources,
        RobertaVocabResources,
    },
    t5::{T5ConfigResources, T5ModelResources, T5VocabResources},
};

/// # Configuration for sentence embeddings
///
/// Contains information regarding the transformer model to load, the optional extra
/// layers, and device to place the model on.
pub struct SentenceEmbeddingsConfig {
    /// Modules configuration resource, contains layers definition
    pub modules_config_resource: Box<dyn ResourceProvider + Send>,
    /// Transformer model type
    pub transformer_type: ModelType,
    /// Transformer model configuration resource
    pub transformer_config_resource: Box<dyn ResourceProvider + Send>,
    /// Transformer weights resource
    pub transformer_weights_resource: Box<dyn ResourceProvider + Send>,
    /// Pooling layer configuration resource
    pub pooling_config_resource: Box<dyn ResourceProvider + Send>,
    /// Optional dense layer configuration resource
    pub dense_config_resource: Option<Box<dyn ResourceProvider + Send>>,
    /// Optional dense layer weights resource
    pub dense_weights_resource: Option<Box<dyn ResourceProvider + Send>>,
    /// Sentence BERT specific configuration resource
    pub sentence_bert_config_resource: Box<dyn ResourceProvider + Send>,
    /// Transformer's tokenizer configuration resource
    pub tokenizer_config_resource: Box<dyn ResourceProvider + Send>,
    /// Transformer's tokenizer vocab resource
    pub tokenizer_vocab_resource: Box<dyn ResourceProvider + Send>,
    /// Optional transformer's tokenizer merges resource
    pub tokenizer_merges_resource: Option<Box<dyn ResourceProvider + Send>>,
    /// Device to place the transformer model on
    pub device: Device,
}

#[cfg(feature = "remote")]
impl From<SentenceEmbeddingsModelType> for SentenceEmbeddingsConfig {
    fn from(model_type: SentenceEmbeddingsModelType) -> Self {
        match model_type {
            SentenceEmbeddingsModelType::DistiluseBaseMultilingualCased => SentenceEmbeddingsConfig {
                modules_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsModulesConfigResources::DISTILUSE_BASE_MULTILINGUAL_CASED,
                )),
                transformer_type: ModelType::DistilBert,
                transformer_config_resource: Box::new(RemoteResource::from_pretrained(
                    DistilBertConfigResources::DISTILUSE_BASE_MULTILINGUAL_CASED,
                )),
                transformer_weights_resource: Box::new(RemoteResource::from_pretrained(
                    DistilBertModelResources::DISTILUSE_BASE_MULTILINGUAL_CASED,
                )),
                pooling_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsPoolingConfigResources::DISTILUSE_BASE_MULTILINGUAL_CASED,
                )),
                dense_config_resource: Some(Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsDenseConfigResources::DISTILUSE_BASE_MULTILINGUAL_CASED,
                ))),
                dense_weights_resource: Some(Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsDenseResources::DISTILUSE_BASE_MULTILINGUAL_CASED,
                ))),
                sentence_bert_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsConfigResources::DISTILUSE_BASE_MULTILINGUAL_CASED,
                )),
                tokenizer_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsTokenizerConfigResources::DISTILUSE_BASE_MULTILINGUAL_CASED,
                )),
                tokenizer_vocab_resource: Box::new(RemoteResource::from_pretrained(
                    DistilBertVocabResources::DISTILUSE_BASE_MULTILINGUAL_CASED,
                )),
                tokenizer_merges_resource: None,
                device: Device::cuda_if_available(),
            },

            SentenceEmbeddingsModelType::BertBaseNliMeanTokens => SentenceEmbeddingsConfig {
                modules_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsModulesConfigResources::BERT_BASE_NLI_MEAN_TOKENS,
                )),
                transformer_type: ModelType::Bert,
                transformer_config_resource: Box::new(RemoteResource::from_pretrained(
                    BertConfigResources::BERT_BASE_NLI_MEAN_TOKENS,
                )),
                transformer_weights_resource: Box::new(RemoteResource::from_pretrained(
                    BertModelResources::BERT_BASE_NLI_MEAN_TOKENS,
                )),
                pooling_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsPoolingConfigResources::BERT_BASE_NLI_MEAN_TOKENS,
                )),
                dense_config_resource: None,
                dense_weights_resource: None,
                sentence_bert_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsConfigResources::BERT_BASE_NLI_MEAN_TOKENS,
                )),
                tokenizer_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsTokenizerConfigResources::BERT_BASE_NLI_MEAN_TOKENS,
                )),
                tokenizer_vocab_resource: Box::new(RemoteResource::from_pretrained(
                    BertVocabResources::BERT_BASE_NLI_MEAN_TOKENS,
                )),
                tokenizer_merges_resource: None,
                device: Device::cuda_if_available(),
            },

            SentenceEmbeddingsModelType::AllMiniLmL12V2 => SentenceEmbeddingsConfig {
                modules_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsModulesConfigResources::ALL_MINI_LM_L12_V2,
                )),
                transformer_type: ModelType::Bert,
                transformer_config_resource: Box::new(RemoteResource::from_pretrained(
                    BertConfigResources::ALL_MINI_LM_L12_V2,
                )),
                transformer_weights_resource: Box::new(RemoteResource::from_pretrained(
                    BertModelResources::ALL_MINI_LM_L12_V2,
                )),
                pooling_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsPoolingConfigResources::ALL_MINI_LM_L12_V2,
                )),
                dense_config_resource: None,
                dense_weights_resource: None,
                sentence_bert_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsConfigResources::ALL_MINI_LM_L12_V2,
                )),
                tokenizer_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsTokenizerConfigResources::ALL_MINI_LM_L12_V2,
                )),
                tokenizer_vocab_resource: Box::new(RemoteResource::from_pretrained(
                    BertVocabResources::ALL_MINI_LM_L12_V2,
                )),
                tokenizer_merges_resource: None,
                device: Device::cuda_if_available(),
            },

            SentenceEmbeddingsModelType::AllMiniLmL6V2 => SentenceEmbeddingsConfig {
                modules_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsModulesConfigResources::ALL_MINI_LM_L6_V2,
                )),
                transformer_type: ModelType::Bert,
                transformer_config_resource: Box::new(RemoteResource::from_pretrained(
                    BertConfigResources::ALL_MINI_LM_L6_V2,
                )),
                transformer_weights_resource: Box::new(RemoteResource::from_pretrained(
                    BertModelResources::ALL_MINI_LM_L6_V2,
                )),
                pooling_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsPoolingConfigResources::ALL_MINI_LM_L6_V2,
                )),
                dense_config_resource: None,
                dense_weights_resource: None,
                sentence_bert_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsConfigResources::ALL_MINI_LM_L6_V2,
                )),
                tokenizer_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsTokenizerConfigResources::ALL_MINI_LM_L6_V2,
                )),
                tokenizer_vocab_resource: Box::new(RemoteResource::from_pretrained(
                    BertVocabResources::ALL_MINI_LM_L6_V2,
                )),
                tokenizer_merges_resource: None,
                device: Device::cuda_if_available(),
            },

            SentenceEmbeddingsModelType::AllDistilrobertaV1 => SentenceEmbeddingsConfig {
                modules_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsModulesConfigResources::ALL_DISTILROBERTA_V1,
                )),
                transformer_type: ModelType::Roberta,
                transformer_config_resource: Box::new(RemoteResource::from_pretrained(
                    RobertaConfigResources::ALL_DISTILROBERTA_V1,
                )),
                transformer_weights_resource: Box::new(RemoteResource::from_pretrained(
                    RobertaModelResources::ALL_DISTILROBERTA_V1,
                )),
                pooling_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsPoolingConfigResources::ALL_DISTILROBERTA_V1,
                )),
                dense_config_resource: None,
                dense_weights_resource: None,
                sentence_bert_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsConfigResources::ALL_DISTILROBERTA_V1,
                )),
                tokenizer_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsTokenizerConfigResources::ALL_DISTILROBERTA_V1,
                )),
                tokenizer_vocab_resource: Box::new(RemoteResource::from_pretrained(
                    RobertaVocabResources::ALL_DISTILROBERTA_V1,
                )),
                tokenizer_merges_resource: Some(Box::new(RemoteResource::from_pretrained(
                    RobertaMergesResources::ALL_DISTILROBERTA_V1,
                ))),
                device: Device::cuda_if_available(),
            },

            SentenceEmbeddingsModelType::ParaphraseAlbertSmallV2 => SentenceEmbeddingsConfig {
                modules_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsModulesConfigResources::PARAPHRASE_ALBERT_SMALL_V2,
                )),
                transformer_type: ModelType::Albert,
                transformer_config_resource: Box::new(RemoteResource::from_pretrained(
                    AlbertConfigResources::PARAPHRASE_ALBERT_SMALL_V2,
                )),
                transformer_weights_resource: Box::new(RemoteResource::from_pretrained(
                    AlbertModelResources::PARAPHRASE_ALBERT_SMALL_V2,
                )),
                pooling_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsPoolingConfigResources::PARAPHRASE_ALBERT_SMALL_V2,
                )),
                dense_config_resource: None,
                dense_weights_resource: None,
                sentence_bert_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsConfigResources::PARAPHRASE_ALBERT_SMALL_V2,
                )),
                tokenizer_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsTokenizerConfigResources::PARAPHRASE_ALBERT_SMALL_V2,
                )),
                tokenizer_vocab_resource: Box::new(RemoteResource::from_pretrained(
                    AlbertVocabResources::PARAPHRASE_ALBERT_SMALL_V2,
                )),
                tokenizer_merges_resource: None,
                device: Device::cuda_if_available(),
            },

            SentenceEmbeddingsModelType::SentenceT5Base => SentenceEmbeddingsConfig {
                modules_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsModulesConfigResources::SENTENCE_T5_BASE,
                )),
                transformer_type: ModelType::T5,
                transformer_config_resource: Box::new(RemoteResource::from_pretrained(
                    T5ConfigResources::SENTENCE_T5_BASE,
                )),
                transformer_weights_resource: Box::new(RemoteResource::from_pretrained(
                    T5ModelResources::SENTENCE_T5_BASE,
                )),
                pooling_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsPoolingConfigResources::SENTENCE_T5_BASE,
                )),
                dense_config_resource: Some(Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsDenseConfigResources::SENTENCE_T5_BASE,
                ))),
                dense_weights_resource: Some(Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsDenseResources::SENTENCE_T5_BASE,
                ))),
                sentence_bert_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsConfigResources::SENTENCE_T5_BASE,
                )),
                tokenizer_config_resource: Box::new(RemoteResource::from_pretrained(
                    SentenceEmbeddingsTokenizerConfigResources::SENTENCE_T5_BASE,
                )),
                tokenizer_vocab_resource: Box::new(RemoteResource::from_pretrained(
                    T5VocabResources::SENTENCE_T5_BASE,
                )),
                tokenizer_merges_resource: None,
                device: Device::cuda_if_available(),
            },
        }
    }
}

/// Configuration for the modules that define the model's layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceEmbeddingsModulesConfig(pub Vec<SentenceEmbeddingsModuleConfig>);

impl std::ops::Deref for SentenceEmbeddingsModulesConfig {
    type Target = Vec<SentenceEmbeddingsModuleConfig>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for SentenceEmbeddingsModulesConfig {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Vec<SentenceEmbeddingsModuleConfig>> for SentenceEmbeddingsModulesConfig {
    fn from(source: Vec<SentenceEmbeddingsModuleConfig>) -> Self {
        Self(source)
    }
}

impl Config for SentenceEmbeddingsModulesConfig {}

impl SentenceEmbeddingsModulesConfig {
    pub fn validate(self) -> Result<Self, RustBertError> {
        match self.get(0) {
            Some(SentenceEmbeddingsModuleConfig {
                module_type: SentenceEmbeddingsModuleType::Transformer,
                ..
            }) => (),
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
        }

        match self.get(1) {
            Some(SentenceEmbeddingsModuleConfig {
                module_type: SentenceEmbeddingsModuleType::Pooling,
                ..
            }) => (),
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
        }

        Ok(self)
    }

    pub fn transformer_module(&self) -> &SentenceEmbeddingsModuleConfig {
        self.get(0).as_ref().unwrap()
    }

    pub fn pooling_module(&self) -> &SentenceEmbeddingsModuleConfig {
        self.get(1).as_ref().unwrap()
    }

    pub fn dense_module(&self) -> Option<&SentenceEmbeddingsModuleConfig> {
        for i in 2..=3 {
            if let Some(SentenceEmbeddingsModuleConfig {
                module_type: SentenceEmbeddingsModuleType::Dense,
                ..
            }) = self.get(i)
            {
                return self.get(i);
            }
        }
        None
    }

    pub fn has_normalization(&self) -> bool {
        for i in 2..=3 {
            if let Some(SentenceEmbeddingsModuleConfig {
                module_type: SentenceEmbeddingsModuleType::Normalize,
                ..
            }) = self.get(i)
            {
                return true;
            }
        }
        false
    }
}

/// Configuration defining a single module (model's layer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceEmbeddingsModuleConfig {
    pub idx: usize,
    pub name: String,
    pub path: String,
    #[serde(rename = "type")]
    #[serde(with = "serde_sentence_embeddings_module_type")]
    pub module_type: SentenceEmbeddingsModuleType,
}

/// Available module types, based on Sentence-Transformers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SentenceEmbeddingsModuleType {
    Transformer,
    Pooling,
    Dense,
    Normalize,
}

mod serde_sentence_embeddings_module_type {
    use super::SentenceEmbeddingsModuleType;
    use serde::{de, Deserializer, Serializer};

    pub fn serialize<S>(
        module_type: &SentenceEmbeddingsModuleType,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("sentence_transformers.models.{:?}", module_type))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SentenceEmbeddingsModuleType, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SentenceEmbeddingsModuleTypeVisitor;

        impl de::Visitor<'_> for SentenceEmbeddingsModuleTypeVisitor {
            type Value = SentenceEmbeddingsModuleType;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a sentence embeddings module type")
            }

            fn visit_str<E: de::Error>(self, s: &str) -> Result<Self::Value, E> {
                s.split('.')
                    .last()
                    .map(|s| serde_json::from_value(serde_json::Value::String(s.to_string())))
                    .transpose()
                    .map_err(de::Error::custom)?
                    .ok_or_else(|| format!("Invalid SentenceEmbeddingsModuleType: {}", s))
                    .map_err(de::Error::custom)
            }
        }

        deserializer.deserialize_str(SentenceEmbeddingsModuleTypeVisitor)
    }
}

/// Configuration for Sentence-Transformers specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceEmbeddingsSentenceBertConfig {
    pub max_seq_length: usize,
    pub do_lower_case: bool,
}

impl Config for SentenceEmbeddingsSentenceBertConfig {}

/// Configuration for transformer's tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceEmbeddingsTokenizerConfig {
    pub add_prefix_space: Option<bool>,
    pub strip_accents: Option<bool>,
    pub do_lower_case: Option<bool>,
}

impl Config for SentenceEmbeddingsTokenizerConfig {}
