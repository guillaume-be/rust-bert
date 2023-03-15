use crate::pipelines::generation_utils::{Cache, LMModelOutput};
use crate::pipelines::onnx::config::{
    ONNXEnvironmentConfig, ATTENTION_MASK_NAME, ENCODER_ATTENTION_MASK_NAME,
    ENCODER_HIDDEN_STATES_NAME, INPUT_IDS_NAME,
};
use crate::pipelines::onnx::conversion::{ort_tensor_to_tch, tch_tensor_to_ort, ONNXLayerCache};
use crate::RustBertError;
use ort::{Environment, Session};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tch::Tensor;

#[derive(Debug)]
pub struct DecoderNameMapping {
    pub decoder_input_names: Vec<String>,
    pub decoder_output_names: HashMap<String, usize>,
    pub decoder_key_value_output_names: HashMap<String, usize>,
}

pub fn get_input_output_mapping(session: &Session) -> DecoderNameMapping {
    let decoder_input_names = session
        .inputs
        .iter()
        .map(|input| input.name.clone())
        .collect::<Vec<String>>();

    let decoder_output_names = session
        .outputs
        .iter()
        .enumerate()
        .map(|(pos, output)| (output.name.clone(), pos))
        .collect::<HashMap<String, usize>>();

    let decoder_key_value_output_names = decoder_output_names
        .iter()
        .filter(|(name, _)| name.contains(".key") | name.contains(".value"))
        .map(|(name, pos)| (name.clone(), *pos))
        .collect::<HashMap<String, usize>>();

    DecoderNameMapping {
        decoder_input_names,
        decoder_output_names,
        decoder_key_value_output_names,
    }
}

pub struct ONNXDecoder {
    session: Session,
    name_mapping: DecoderNameMapping,
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

        let inputs = self
            .name_mapping
            .decoder_input_names
            .iter()
            .map(|input_name| {
                if let Some(tensor) = input_dict.remove(input_name.as_str()) {
                    Ok(tch_tensor_to_ort(tensor)?)
                } else {
                    tch_tensor_to_ort(
                        layer_states
                            .ok_or_else(|| {
                                return RustBertError::OrtError(format!(
                                    "{input_name} not found and cache was not provided."
                                ));
                            })?
                            .values
                            .get(&input_name.replace("past_key_values", "present"))
                            .unwrap(),
                    )
                }
            })
            .collect::<Result<Vec<_>, RustBertError>>()?;

        let outputs = self.session.run(inputs)?;

        let lm_logits = ort_tensor_to_tch(
            &outputs[*self
                .name_mapping
                .decoder_output_names
                .get("logits")
                .unwrap()],
        )?;
        let cache = if self.use_cache {
            Cache::ONNXCache(ONNXLayerCache::from_ort_output(
                &outputs,
                &self.name_mapping.decoder_key_value_output_names,
            )?)
        } else {
            Cache::None
        };

        Ok(LMModelOutput { lm_logits, cache })
    }
}
