use serde::{Deserialize, Serialize};
use tch::Device;

use crate::bert::{BertConfigResources, BertModelResources, BertVocabResources};
use crate::pipelines::common::ModelType;
use crate::pipelines::sentence_embeddings::{
    SentenceEmbeddingsModulesConfigResources, SentenceEmbeddingsPoolingConfigResources,
    SentenceEmbeddingsTokenizerConfigResources,
};
use crate::resources::{RemoteResource, ResourceProvider};
use crate::{Config, RustBertError};

pub struct SentenceEmbeddingsConfig {
    pub modules_config_resource: Box<dyn ResourceProvider + Send>,
    pub transformer_type: ModelType,
    pub transformer_config_resource: Box<dyn ResourceProvider + Send>,
    pub transformer_weights_resource: Box<dyn ResourceProvider + Send>,
    pub pooling_config_resource: Box<dyn ResourceProvider + Send>,
    pub dense_config_resource: Option<Box<dyn ResourceProvider + Send>>,
    pub dense_weights_resource: Option<Box<dyn ResourceProvider + Send>>,
    pub tokenizer_config_resource: Box<dyn ResourceProvider + Send>,
    pub tokenizer_vocab_resource: Box<dyn ResourceProvider + Send>,
    pub tokenizer_merges_resource: Option<Box<dyn ResourceProvider + Send>>,
    pub device: Device,
}

impl Default for SentenceEmbeddingsConfig {
    fn default() -> Self {
        SentenceEmbeddingsConfig {
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
            tokenizer_config_resource: Box::new(RemoteResource::from_pretrained(
                SentenceEmbeddingsTokenizerConfigResources::ALL_MINI_LM_L12_V2,
            )),
            tokenizer_vocab_resource: Box::new(RemoteResource::from_pretrained(
                BertVocabResources::ALL_MINI_LM_L12_V2,
            )),
            tokenizer_merges_resource: None,
            device: Device::cuda_if_available(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceEmbeddingsModules(pub Vec<SentenceEmbeddingsModule>);

impl std::ops::Deref for SentenceEmbeddingsModules {
    type Target = Vec<SentenceEmbeddingsModule>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for SentenceEmbeddingsModules {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Vec<SentenceEmbeddingsModule>> for SentenceEmbeddingsModules {
    fn from(source: Vec<SentenceEmbeddingsModule>) -> Self {
        Self(source)
    }
}

impl Config for SentenceEmbeddingsModules {}

impl SentenceEmbeddingsModules {
    pub fn validate(self) -> Result<Self, RustBertError> {
        match self.get(0) {
            Some(SentenceEmbeddingsModule {
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
            Some(SentenceEmbeddingsModule {
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

    pub fn transformer_module(&self) -> &SentenceEmbeddingsModule {
        self.get(0).as_ref().unwrap()
    }

    pub fn pooling_module(&self) -> &SentenceEmbeddingsModule {
        self.get(1).as_ref().unwrap()
    }

    pub fn dense_module(&self) -> Option<&SentenceEmbeddingsModule> {
        for i in 2..=3 {
            if let Some(SentenceEmbeddingsModule {
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
            if let Some(SentenceEmbeddingsModule {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceEmbeddingsModule {
    pub idx: usize,
    pub name: String,
    pub path: String,
    #[serde(rename = "type")]
    #[serde(with = "serde_sentence_embeddings_module_type")]
    pub module_type: SentenceEmbeddingsModuleType,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceEmbeddingsTokenizerConfig {
    pub max_seq_length: usize,
    pub do_lower_case: bool,
}

impl Config for SentenceEmbeddingsTokenizerConfig {}
