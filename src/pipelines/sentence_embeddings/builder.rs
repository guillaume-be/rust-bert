use std::path::PathBuf;

use serde::Deserialize;
use tch::Device;

use crate::pipelines::common::ModelType;
use crate::pipelines::sentence_embeddings::{
    SentenceEmbeddingsConfig, SentenceEmbeddingsModel, SentenceEmbeddingsModules,
};
use crate::{Config, RustBertError};

pub struct SentenceEmbeddingsBuilder<T> {
    device: Option<Device>,
    inner: T,
}

impl<T> SentenceEmbeddingsBuilder<T> {
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }
}

impl SentenceEmbeddingsBuilder<Local> {
    pub fn local<P: Into<PathBuf>>(model_dir: P) -> Self {
        Self {
            device: None,
            inner: Local {
                model_dir: model_dir.into(),
            },
        }
    }

    pub fn create_model(self) -> Result<SentenceEmbeddingsModel, RustBertError> {
        let model_dir = self.inner.model_dir;

        let modules_config = model_dir.join("modules.json");
        let modules = SentenceEmbeddingsModules::from_file(&modules_config).validate()?;

        let transformer_config = model_dir.join("config.json");
        let transformer_type = ModelConfig::from_file(&transformer_config).model_type;
        let transformer_weights = model_dir.join("rust_model.ot");

        let pooling_config = model_dir
            .join(&modules.pooling_module().path)
            .join("config.json");

        let (dense_config, dense_weights) = modules
            .dense_module()
            .map(|m| {
                (
                    Some(model_dir.join(&m.path).join("config.json")),
                    Some(model_dir.join(&m.path).join("rust_model.ot")),
                )
            })
            .unwrap_or((None, None));

        let tokenizer_config = model_dir.join("sentence_bert_config.json");
        let (tokenizer_vocab, tokenizer_merges) = match transformer_type {
            ModelType::Bert | ModelType::DistilBert => (model_dir.join("vocab.txt"), None),
            ModelType::Roberta => (
                model_dir.join("vocab.json"),
                Some(model_dir.join("merges.txt")),
            ),
            ModelType::Albert => (model_dir.join("spiece.model"), None),
            ModelType::T5 => (model_dir.join("spiece.model"), None),
            _ => {
                return Err(RustBertError::InvalidConfigurationError(format!(
                    "Unsupported transformer model {:?} for Sentence Embeddings",
                    transformer_type
                )));
            }
        };

        let device = self.device.ok_or_else(|| {
            RustBertError::InvalidConfigurationError("Missing device configuration".into())
        })?;

        let config = SentenceEmbeddingsConfig {
            modules_config_resource: modules_config.into(),
            transformer_type,
            transformer_config_resource: transformer_config.into(),
            transformer_weights_resource: transformer_weights.into(),
            pooling_config_resource: pooling_config.into(),
            dense_config_resource: dense_config.map(|r| r.into()),
            dense_weights_resource: dense_weights.map(|r| r.into()),
            tokenizer_config_resource: tokenizer_config.into(),
            tokenizer_vocab_resource: tokenizer_vocab.into(),
            tokenizer_merges_resource: tokenizer_merges.map(|r| r.into()),
            device,
        };

        SentenceEmbeddingsModel::new(config)
    }
}

pub struct Local {
    model_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    model_type: ModelType,
}

impl Config for ModelConfig {}
