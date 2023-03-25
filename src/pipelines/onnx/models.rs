use crate::pipelines::common::{ConfigOption, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::{
    PreparedInput, PrivateLanguageGenerator,
};
use crate::pipelines::generation_utils::{Cache, GenerateConfig, LMModelOutput, LanguageGenerator};
use crate::pipelines::onnx::config::ONNXEnvironmentConfig;
use crate::pipelines::onnx::decoder::ONNXDecoder;
use crate::pipelines::onnx::encoder::ONNXEncoder;
use crate::{Config, RustBertError};

use ort::{Environment, ExecutionProvider};
use rust_tokenizers::tokenizer::TruncationStrategy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tch::{nn, Device, Kind, Tensor};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ONNXModelConfig {
    pub bos_token_id: Option<i64>,
    pub eos_token_ids: Option<Vec<i64>>,
    pub pad_token_id: Option<i64>,
    pub vocab_size: i64,
    pub decoder_start_token_id: Option<i64>,
    pub max_position_embeddings: Option<i64>,
    pub id2label: Option<HashMap<i64, String>>,
}

impl Config for ONNXModelConfig {}

pub struct ONNXCausalGenerator {
    decoder_without_past: Option<ONNXDecoder>,
    decoder_with_past: Option<ONNXDecoder>,
    generate_config: GenerateConfig,
    tokenizer: TokenizerOption,
    bos_token_id: Option<i64>,
    eos_token_ids: Option<Vec<i64>>,
    pad_token_id: Option<i64>,
    is_encoder_decoder: bool,
    vocab_size: i64,
    decoder_start_id: Option<i64>,
    max_position_embeddings: Option<i64>,
    use_past: bool,
}

impl ONNXCausalGenerator {
    pub fn new(
        generate_config: GenerateConfig,
        environment: Option<&Arc<Environment>>,
        onnx_config: Option<&ONNXEnvironmentConfig>,
    ) -> Result<Self, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;
        let merges_path = generate_config
            .merges_resource
            .as_ref()
            .map(|r| r.get_local_path())
            .transpose()?;

        let tokenizer = TokenizerOption::from_file(
            generate_config.model_type,
            vocab_path.to_str().unwrap(),
            merges_path.as_ref().and_then(|path| path.to_str()),
            false,
            None,
            None,
        )?;

        Self::new_with_tokenizer(generate_config, tokenizer, environment, onnx_config)
    }

    pub fn new_with_tokenizer(
        generate_config: GenerateConfig,
        tokenizer: TokenizerOption,
        environment: Option<&Arc<Environment>>,
        onnx_config: Option<&ONNXEnvironmentConfig>,
    ) -> Result<Self, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let model_config = ConfigOption::from_file(generate_config.model_type, config_path);

        let (_, decoder_without_past_file, decoder_with_past_file) =
            generate_config.model_resource.get_onnx_local_paths()?;

        if decoder_without_past_file.is_none() & decoder_with_past_file.is_none() {
            return Err(RustBertError::InvalidConfigurationError("Must provide at least one of `decoder_without_past_file`, `decoder_with_past_file`, both set to None".to_string()));
        }

        let default_onnx_config = if onnx_config.is_none() {
            let mut execution_providers = Vec::new();
            if let Device::Cuda(_) = generate_config.device {
                execution_providers.push(ExecutionProvider::cuda());
            };
            execution_providers.push(ExecutionProvider::cpu());
            Some(ONNXEnvironmentConfig {
                execution_providers: Some(execution_providers),
                ..Default::default()
            })
        } else {
            None
        };
        let onnx_config = onnx_config.unwrap_or_else(|| &default_onnx_config.as_ref().unwrap());

        let local_environment = if environment.is_none() {
            Some(Arc::new(
                Environment::builder()
                    .with_name("ONNXConditionalGenerator environment")
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

        let bos_token_id = tokenizer.get_bos_id();
        let eos_token_ids = tokenizer.get_eos_id().map(|id| vec![id]);
        let pad_token_id = tokenizer.get_pad_id();
        let max_position_embeddings = model_config.get_max_len();
        let is_encoder_decoder = false;
        let vocab_size = model_config.get_vocab_size();
        let decoder_start_id = model_config.get_decoder_start_token_id();
        let use_past = decoder_with_past.is_some();

        Ok(Self {
            decoder_without_past,
            decoder_with_past,
            generate_config,
            tokenizer,
            bos_token_id,
            eos_token_ids,
            pad_token_id,
            is_encoder_decoder,
            vocab_size,
            decoder_start_id,
            max_position_embeddings,
            use_past,
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
                    "Invalid cache type provided, expected Cache::ONNXLayerCache, got {:?}.",
                    cache
                )));
            }
        }
    }
}

impl PrivateLanguageGenerator for ONNXCausalGenerator {
    fn _get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
    }
    fn get_device(&self) -> Device {
        Device::Cpu
    }
    fn get_var_store_mut(&mut self) -> Result<&mut nn::VarStore, RustBertError> {
        Err(RustBertError::ValueError(
            "No VarStore available for ONNX models".to_string(),
        ))
    }
    fn get_config(&self) -> &GenerateConfig {
        &self.generate_config
    }
    fn get_bos_id(&self) -> Option<i64> {
        self.bos_token_id
    }
    fn get_eos_ids(&self) -> Option<&Vec<i64>> {
        self.eos_token_ids.as_ref()
    }
    fn get_pad_id(&self) -> Option<i64> {
        self.pad_token_id
    }
    fn is_encoder_decoder(&self) -> bool {
        self.is_encoder_decoder
    }
    fn get_vocab_size(&self) -> i64 {
        self.vocab_size
    }
    fn get_decoder_start_id(&self) -> Option<i64> {
        self.decoder_start_id
    }
    fn get_max_positions_embeddings(&self) -> Option<i64> {
        self.max_position_embeddings
    }

    fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        layer_past: Cache,
        attention_mask: Option<&Tensor>,
        _token_type_ids: Option<&Tensor>,
        _position_ids: Option<&Tensor>,
        _input_embeds: Option<&Tensor>,
        _encoder_outputs: Option<&Tensor>,
        _decoder_input_ids: Option<&Tensor>,
        _train: bool,
    ) -> Result<LMModelOutput, RustBertError> {
        self.forward(input_ids, attention_mask, None, None, Some(&layer_past))
    }

    fn prepare_inputs_for_generation<'a>(
        &self,
        input_ids: Tensor,
        _encoder_outputs: Option<&'a Tensor>,
        past: Cache,
        attention_mask: Tensor,
    ) -> PreparedInput<'a> {
        let position_ids = (attention_mask.totype(Kind::Int64).cumsum(-1, Kind::Int64) - 1)
            .masked_fill(&attention_mask.eq(0), 1);

        match (past, self.use_past) {
            (Cache::ONNXCache(past), true) => PreparedInput {
                prepared_input: Some(input_ids.select(1, -1).unsqueeze(-1)),
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: None,
                prepared_decoder_input: None,
                prepared_position_ids: Some(position_ids.select(1, -1).unsqueeze(-1)),
                prepared_past: Cache::ONNXCache(past),
            },
            _ => PreparedInput {
                prepared_input: Some(input_ids),
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: None,
                prepared_decoder_input: None,
                prepared_position_ids: Some(position_ids),
                prepared_past: Cache::None,
            },
        }
    }

    fn reorder_cache(
        &self,
        past: &mut Cache,
        _encoder_outputs: Option<Tensor>,
        beam_indices: &Tensor,
    ) -> Option<Tensor> {
        match past {
            Cache::ONNXCache(cached_decoder_state) => {
                for (_, layer_past) in cached_decoder_state.values.iter_mut() {
                    *layer_past = layer_past.index_select(0, beam_indices);
                }
                None
            }
            Cache::None => None,
            _ => {
                panic!("Invalid cache for ONNX model");
            }
        }
    }
}

impl LanguageGenerator for ONNXCausalGenerator {}

pub struct ONNXConditionalGenerator {
    encoder: ONNXEncoder,
    decoder_without_past: Option<ONNXDecoder>,
    decoder_with_past: Option<ONNXDecoder>,
    generate_config: GenerateConfig,
    tokenizer: TokenizerOption,
    bos_token_id: Option<i64>,
    eos_token_ids: Option<Vec<i64>>,
    pad_token_id: Option<i64>,
    is_encoder_decoder: bool,
    vocab_size: i64,
    decoder_start_id: Option<i64>,
    max_position_embeddings: Option<i64>,
    use_past: bool,
}

impl ONNXConditionalGenerator {
    pub fn new(
        generate_config: GenerateConfig,
        environment: Option<&Arc<Environment>>,
        onnx_config: Option<&ONNXEnvironmentConfig>,
    ) -> Result<Self, RustBertError> {
        let vocab_path = generate_config.vocab_resource.get_local_path()?;
        let merges_path = generate_config
            .merges_resource
            .as_ref()
            .map(|r| r.get_local_path())
            .transpose()?;

        let tokenizer = TokenizerOption::from_file(
            generate_config.model_type,
            vocab_path.to_str().unwrap(),
            merges_path.as_ref().and_then(|path| path.to_str()),
            false,
            None,
            None,
        )?;

        Self::new_with_tokenizer(generate_config, tokenizer, environment, onnx_config)
    }

    pub fn new_with_tokenizer(
        generate_config: GenerateConfig,
        tokenizer: TokenizerOption,
        environment: Option<&Arc<Environment>>,
        onnx_config: Option<&ONNXEnvironmentConfig>,
    ) -> Result<Self, RustBertError> {
        let config_path = generate_config.config_resource.get_local_path()?;
        let model_config = ConfigOption::from_file(generate_config.model_type, config_path);

        let (encoder_file, decoder_without_past_file, decoder_with_past_file) =
            generate_config.model_resource.get_onnx_local_paths()?;

        if decoder_without_past_file.is_none() & decoder_with_past_file.is_none() {
            return Err(RustBertError::InvalidConfigurationError("Must provide at least one of `decoder_without_past_file`, `decoder_with_past_file`, both set to None".to_string()));
        }

        let default_onnx_config = if onnx_config.is_none() {
            let mut execution_providers = Vec::new();
            if let Device::Cuda(_) = generate_config.device {
                execution_providers.push(ExecutionProvider::cuda());
            };
            execution_providers.push(ExecutionProvider::cpu());
            Some(ONNXEnvironmentConfig {
                execution_providers: Some(execution_providers),
                ..Default::default()
            })
        } else {
            None
        };
        let onnx_config = onnx_config.unwrap_or_else(|| &default_onnx_config.as_ref().unwrap());

        let local_environment = if environment.is_none() {
            Some(Arc::new(
                Environment::builder()
                    .with_name("ONNXConditionalGenerator environment")
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
        let encoder_file = encoder_file.ok_or_else(|| {return
            RustBertError::InvalidConfigurationError(format!("ONNXConditionalGenerator requires an `enoder_path` to be provided in the `ModelResources`, got {:?}", generate_config.model_resource))})?;

        let encoder = ONNXEncoder::new(encoder_file, environment, onnx_config)?;
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

        let bos_token_id = tokenizer.get_bos_id();
        let eos_token_ids = tokenizer.get_eos_id().map(|id| vec![id]);
        let pad_token_id = tokenizer.get_pad_id();
        let max_position_embeddings = model_config.get_max_len();
        let is_encoder_decoder = true;
        let vocab_size = model_config.get_vocab_size();
        let decoder_start_id = model_config.get_decoder_start_token_id();
        let use_past = decoder_with_past.is_some();

        Ok(Self {
            encoder,
            decoder_without_past,
            decoder_with_past,
            generate_config,
            tokenizer,
            bos_token_id,
            eos_token_ids,
            pad_token_id,
            is_encoder_decoder,
            vocab_size,
            decoder_start_id,
            max_position_embeddings,
            use_past,
        })
    }

    pub fn forward(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        layer_states: Option<&Cache>,
    ) -> Result<LMModelOutput, RustBertError> {
        let calc_encoder_output = if encoder_hidden_states.is_none() {
            Some(
                self.encoder
                    .forward(input_ids, encoder_attention_mask, None, None, None)?
                    .last_hidden_state,
            )
        } else {
            None
        };
        let encoder_hidden_states =
            encoder_hidden_states.unwrap_or_else(|| calc_encoder_output.as_ref().unwrap());

        match (
            &self.decoder_without_past,
            &self.decoder_with_past,
            layer_states,
        ) {
            (Some(ref decoder_without_past), _, None)
            | (Some(ref decoder_without_past), _, Some(Cache::None)) => decoder_without_past
                .forward(
                    decoder_input_ids,
                    attention_mask,
                    Some(encoder_hidden_states),
                    encoder_attention_mask,
                    None,
                ),
            (_, Some(ref decoder_with_past), Some(Cache::ONNXCache(ref onnx_cache))) => {
                decoder_with_past.forward(
                    decoder_input_ids,
                    attention_mask,
                    Some(encoder_hidden_states),
                    encoder_attention_mask,
                    Some(onnx_cache),
                )
            }
            (Some(ref decoder_without_past), None, Some(Cache::ONNXCache(_))) => {
                decoder_without_past.forward(
                    decoder_input_ids,
                    attention_mask,
                    Some(encoder_hidden_states),
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
                    "Invalid cache type provided, expected Cache::ONNXLayerCache, got {:?}.",
                    cache
                )));
            }
        }
    }
}

impl PrivateLanguageGenerator for ONNXConditionalGenerator {
    fn _get_tokenizer(&self) -> &TokenizerOption {
        &self.tokenizer
    }
    fn get_device(&self) -> Device {
        Device::Cpu
    }
    fn get_var_store_mut(&mut self) -> Result<&mut nn::VarStore, RustBertError> {
        Err(RustBertError::ValueError(
            "No VarStore available for ONNX models".to_string(),
        ))
    }
    fn get_config(&self) -> &GenerateConfig {
        &self.generate_config
    }
    fn get_bos_id(&self) -> Option<i64> {
        self.bos_token_id
    }
    fn get_eos_ids(&self) -> Option<&Vec<i64>> {
        self.eos_token_ids.as_ref()
    }
    fn get_pad_id(&self) -> Option<i64> {
        self.pad_token_id
    }
    fn is_encoder_decoder(&self) -> bool {
        self.is_encoder_decoder
    }
    fn get_vocab_size(&self) -> i64 {
        self.vocab_size
    }
    fn get_decoder_start_id(&self) -> Option<i64> {
        self.decoder_start_id
    }
    fn get_max_positions_embeddings(&self) -> Option<i64> {
        self.max_position_embeddings
    }

    fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        layer_past: Cache,
        attention_mask: Option<&Tensor>,
        _token_type_ids: Option<&Tensor>,
        _position_ids: Option<&Tensor>,
        _input_embeds: Option<&Tensor>,
        encoder_outputs: Option<&Tensor>,
        decoder_input_ids: Option<&Tensor>,
        _train: bool,
    ) -> Result<LMModelOutput, RustBertError> {
        self.forward(
            input_ids,
            attention_mask,
            encoder_outputs,
            None,
            decoder_input_ids,
            Some(&layer_past),
        )
    }

    fn encode(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Option<Tensor> {
        Some(
            self.encoder
                .forward(Some(input_ids), attention_mask, None, None, None)
                .unwrap()
                .last_hidden_state,
        )
    }

    fn prepare_inputs_for_generation<'a>(
        &self,
        input_ids: Tensor,
        encoder_outputs: Option<&'a Tensor>,
        past: Cache,
        attention_mask: Tensor,
    ) -> PreparedInput<'a> {
        match (past, self.use_past) {
            (Cache::ONNXCache(past), true) => PreparedInput {
                prepared_input: None,
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: encoder_outputs,
                prepared_decoder_input: Some(input_ids.narrow(1, -1, 1)),
                prepared_position_ids: None,
                prepared_past: Cache::ONNXCache(past),
            },
            _ => PreparedInput {
                prepared_input: None,
                prepared_attention_mask: Some(attention_mask),
                prepared_encoder_output: encoder_outputs,
                prepared_decoder_input: Some(input_ids),
                prepared_position_ids: None,
                prepared_past: Cache::None,
            },
        }
    }

    fn encode_prompt_text<S>(
        &self,
        prompt_text: &[S],
        max_len: Option<i64>,
        pad_token_id: Option<i64>,
    ) -> Tensor
    where
        S: AsRef<str> + Sync,
    {
        let tokens = self.get_tokenizer().encode_list(
            prompt_text,
            max_len
                .map(|max_len| max_len as usize)
                .unwrap_or(usize::MAX),
            &TruncationStrategy::LongestFirst,
            0,
        );
        let token_ids = tokens
            .into_iter()
            .map(|tokenized_input| tokenized_input.token_ids)
            .collect::<Vec<Vec<i64>>>();

        let max_len = token_ids.iter().map(|input| input.len()).max().unwrap();

        let pad_token = match pad_token_id {
            Some(value) => value,
            None => self.get_tokenizer().get_unk_id(),
        };

        let token_ids = token_ids
            .into_iter()
            .map(|mut input| {
                let temp = vec![pad_token; max_len - input.len()];
                input.extend(temp);
                input
            })
            .map(|tokens| Tensor::of_slice(&tokens).to(self.get_device()))
            .collect::<Vec<Tensor>>();

        Tensor::stack(&token_ids, 0)
    }

    fn reorder_cache(
        &self,
        past: &mut Cache,
        encoder_outputs: Option<Tensor>,
        beam_indices: &Tensor,
    ) -> Option<Tensor> {
        let encoder_outputs = encoder_outputs.map(|value| value.index_select(0, beam_indices));
        match past {
            Cache::ONNXCache(cached_decoder_state) => {
                for (_, layer_past) in cached_decoder_state.values.iter_mut() {
                    *layer_past = layer_past.index_select(0, beam_indices);
                }
            }
            Cache::None => {}
            _ => {
                panic!("Invalid cache for ONNX model");
            }
        }
        encoder_outputs
    }
}

impl LanguageGenerator for ONNXConditionalGenerator {}
