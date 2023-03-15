use crate::pipelines::generation_utils::{Cache, GenerateConfig, LMModelOutput};
use crate::pipelines::onnx::config::ONNXEnvironmentConfig;
use crate::pipelines::onnx::decoder::ONNXDecoder;
use crate::RustBertError;
use ort::{Environment, ExecutionProvider};
use std::path::PathBuf;
use std::sync::Arc;
use tch::Tensor;

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

    pub fn forward(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        layer_states: Option<&Cache>,
    ) -> Result<LMModelOutput, RustBertError> {
        match (
            &self.decoder_without_past,
            &self.decoder_with_past,
            layer_states,
        ) {
            (Some(ref decoder_without_past), _, None)
            | (Some(ref decoder_without_past), _, Some(Cache::None)) => decoder_without_past
                .forward(
                    input_ids,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                ),
            (_, Some(ref decoder_with_past), Some(Cache::ONNXCache(ref onnx_cache))) => {
                decoder_with_past.forward(
                    input_ids,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    Some(onnx_cache),
                )
            }
            (Some(ref decoder_without_past), None, Some(Cache::ONNXCache(_))) => {
                decoder_without_past.forward(
                    input_ids,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                )
            }
            (None, _, None) => {
                return Err(RustBertError::ValueError(
                    "No decoder_without_cache loaded and no cache provided.".to_string(),
                ));
            }
            (None, None, _) => {
                return Err(RustBertError::ValueError(
                    "No decoder provided.".to_string(),
                ));
            }
            (_, _, Some(cache)) => {
                return Err(RustBertError::ValueError(format!(
                    "Invalid cache type provided, expected Cache::ONNXlayerCache, got {:?}.",
                    cache
                )));
            }
        }
    }
}
