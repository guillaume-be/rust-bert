use crate::pipelines::generation_utils::GenerateConfig;
use crate::pipelines::onnx::config::ONNXEnvironmentConfig;
use crate::pipelines::onnx::decoder::ONNXDecoder;
use crate::RustBertError;
use ort::{Environment, ExecutionProvider};
use std::path::PathBuf;
use std::sync::Arc;

pub struct ONNXCausalDecoder {
    pub decoder_without_past: Option<ONNXDecoder>,
    pub decoder_with_past: Option<ONNXDecoder>,
    pub generate_config: GenerateConfig,
}

impl ONNXCausalDecoder {
    pub fn new(
        decoder_without_past_file: Option<PathBuf>,
        decoder_with_past_file: Option<PathBuf>,
        onnx_config: &ONNXEnvironmentConfig,
        generate_config: GenerateConfig,
        environment: Option<&Arc<Environment>>,
    ) -> Result<Self, RustBertError> {
        if decoder_without_past_file.is_none() & decoder_with_past_file.is_none() {
            return Err(RustBertError::InvalidConfigurationError("Must provide at least one of `decoder_without_past_file`, `decoder_with_past_file`, both set to None".to_string()));
        }

        let local_environment = if environment.is_none() {
            Some(Arc::new(
                Environment::builder()
                    .with_name("ONNXCausalDecoder environment")
                    .with_execution_providers(
                        onnx_config
                            .execution_providers
                            .clone()
                            .unwrap_or(vec![ExecutionProvider::cpu()]),
                    )
                    .build()?,
            ))
        } else {
            None
        };
        let environment = environment.unwrap_or_else(|| local_environment.as_ref().unwrap());

        let decoder_without_past = if let Some(model_file) = decoder_without_past_file {
            Some(ONNXDecoder::new(
                model_file,
                true,
                environment,
                onnx_config,
            )?)
        } else {
            None
        };

        let decoder_with_past = if let Some(model_file) = decoder_with_past_file {
            Some(ONNXDecoder::new(
                model_file,
                true,
                environment,
                onnx_config,
            )?)
        } else {
            None
        };

        Ok(Self {
            decoder_without_past,
            decoder_with_past,
            generate_config,
        })
    }
}
