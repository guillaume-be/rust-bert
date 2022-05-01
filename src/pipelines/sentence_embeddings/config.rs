use serde::{Deserialize, Serialize};
use tch::Device;

use crate::pipelines::common::ModelType;
use crate::resources::ResourceProvider;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceEmbeddingsModuleConfig {
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
