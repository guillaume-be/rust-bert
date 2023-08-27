use crate::pipelines::generation_utils::{Cache, LMModelOutput};
use crate::pipelines::onnx::common::{get_input_output_mapping, InputOutputNameMapping};
use crate::pipelines::onnx::config::{
    ONNXEnvironmentConfig, ATTENTION_MASK_NAME, ENCODER_ATTENTION_MASK_NAME,
    ENCODER_HIDDEN_STATES_NAME, INPUT_IDS_NAME, POSITION_IDS,
};
use crate::pipelines::onnx::conversion::{array_to_ort, ort_tensor_to_tch, tch_tensor_to_ndarray};
use crate::pipelines::onnx::models::ONNXLayerCache;
use crate::RustBertError;
use ort::{Environment, Session};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tch::Tensor;

pub struct ONNXDecoder {
    session: Session,
    name_mapping: InputOutputNameMapping,
    use_cache: bool,
}

impl ONNXDecoder {
    pub fn new(
        model_file: PathBuf,
        use_cache: bool,
        environment: &Arc<Environment>,
        onnx_config: &ONNXEnvironmentConfig,
    ) -> Result<Self, RustBertError> {
        let session = onnx_config
            .get_session_builder(environment)?
            .with_model_from_file(model_file)?;
        let name_mapping = get_input_output_mapping(&session);
        Ok(Self {
            session,
            name_mapping,
            use_cache,
        })
    }

    pub fn forward(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        layer_states: Option<&ONNXLayerCache>,
    ) -> Result<LMModelOutput, RustBertError> {
        let mut input_dict = HashMap::new();
        if let Some(input_ids) = input_ids {
            input_dict.insert(INPUT_IDS_NAME, input_ids);
        }
        if let Some(attention_mask) = attention_mask {
            input_dict.insert(ATTENTION_MASK_NAME, attention_mask);
        }
        if let Some(encoder_hidden_states) = encoder_hidden_states {
            input_dict.insert(ENCODER_HIDDEN_STATES_NAME, encoder_hidden_states);
        }
        if let Some(encoder_attention_mask) = encoder_attention_mask {
            input_dict.insert(ENCODER_ATTENTION_MASK_NAME, encoder_attention_mask);
        }
        if let Some(position_ids) = position_ids {
            input_dict.insert(POSITION_IDS, position_ids);
        }

        let inputs_arrays = self
            .name_mapping
            .input_names
            .iter()
            .map(|input_name| {
                if let Some(tensor) = input_dict.remove(input_name.as_str()) {
                    tch_tensor_to_ndarray(tensor)
                } else {
                    let layer_states = layer_states.ok_or_else(|| {
                        RustBertError::OrtError(format!(
                            "{input_name} not found and cache was not provided."
                        ))
                    })?;
                    let input_pos = layer_states
                        .values
                        .get(&input_name.replace("past", "present"))
                        .or_else(|| {
                            layer_states
                                .values
                                .get(&input_name.replace("past_key_values", "present"))
                        })
                        .ok_or_else(|| {
                            let found_keys = layer_states.values.keys().collect::<Vec<&String>>();
                            RustBertError::OrtError(format!(
                                "{input_name} not found in cache ({found_keys:?})."
                            ))
                        })?;
                    tch_tensor_to_ndarray(input_pos)
                }
            })
            .collect::<Result<Vec<_>, RustBertError>>()?;

        let input_values = inputs_arrays
            .iter()
            .map(|array| array_to_ort(&self.session, array).unwrap())
            .collect::<Vec<_>>();

        let outputs = self.session.run(input_values)?;

        let lm_logits =
            ort_tensor_to_tch(&outputs[*self.name_mapping.output_names.get("logits").unwrap()])?;
        let cache = if self.use_cache {
            Cache::ONNXCache(ONNXLayerCache::from_ort_output(
                &outputs,
                &self.name_mapping.key_value_output_names,
            )?)
        } else {
            Cache::None
        };

        Ok(LMModelOutput { lm_logits, cache })
    }
}
