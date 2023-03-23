use crate::pipelines::onnx::common::{get_input_output_mapping, InputOutputNameMapping};
use crate::pipelines::onnx::config::{
    ONNXEnvironmentConfig, ATTENTION_MASK_NAME, INPUT_EMBEDS, INPUT_IDS_NAME, LAST_HIDDEN_STATE,
    POSITION_IDS, TOKEN_TYPE_IDS,
};
use crate::pipelines::onnx::conversion::{ort_tensor_to_tch, tch_tensor_to_ort};
use crate::RustBertError;
use ort::{Environment, Session};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tch::Tensor;

pub struct ONNXEncoder {
    session: Session,
    name_mapping: InputOutputNameMapping,
}

impl ONNXEncoder {
    pub fn new(
        model_file: PathBuf,
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
        })
    }

    pub fn forward(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
    ) -> Result<ONNXEncoderModelOutput, RustBertError> {
        let mut input_dict = HashMap::new();
        if let Some(input_ids) = input_ids {
            input_dict.insert(INPUT_IDS_NAME, input_ids);
        }
        if let Some(attention_mask) = attention_mask {
            input_dict.insert(ATTENTION_MASK_NAME, attention_mask);
        }
        if let Some(token_type_ids) = token_type_ids {
            input_dict.insert(TOKEN_TYPE_IDS, token_type_ids);
        }
        if let Some(position_ids) = position_ids {
            input_dict.insert(POSITION_IDS, position_ids);
        }
        if let Some(input_embeds) = input_embeds {
            input_dict.insert(INPUT_EMBEDS, input_embeds);
        }

        let inputs = self
            .name_mapping
            .input_names
            .iter()
            .map(|input_name| {
                if let Some(tensor) = input_dict.remove(input_name.as_str()) {
                    tch_tensor_to_ort(tensor)
                } else {
                    Err(RustBertError::OrtError(format!(
                        "{input_name} not found but expected by model."
                    )))
                }
            })
            .collect::<Result<Vec<_>, RustBertError>>()?;

        let outputs = self.session.run(inputs)?;

        let last_hidden_state = ort_tensor_to_tch(
            &outputs[*self
                .name_mapping
                .output_names
                .get(LAST_HIDDEN_STATE)
                .unwrap()],
        )?;
        let (hidden_states, attentions) = if self.name_mapping.output_names.len() > 1 {
            let hidden_states = self
                .name_mapping
                .output_names
                .iter()
                .filter(|(name, _)| name.contains("hidden_states"))
                .map(|(_, position)| outputs.get(*position))
                .map(|array| array.map(|array_value| ort_tensor_to_tch(array_value).unwrap()))
                .collect::<Option<Vec<_>>>();

            let attentions = self
                .name_mapping
                .output_names
                .iter()
                .filter(|(name, _)| name.contains("attentions"))
                .map(|(_, position)| outputs.get(*position))
                .map(|array| array.map(|array_value| ort_tensor_to_tch(array_value).unwrap()))
                .collect::<Option<Vec<_>>>();
            (hidden_states, attentions)
        } else {
            (None, None)
        };
        Ok(ONNXEncoderModelOutput {
            last_hidden_state,
            hidden_states,
            attentions,
        })
    }
}

pub struct ONNXEncoderModelOutput {
    pub last_hidden_state: Tensor,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attentions: Option<Vec<Tensor>>,
}
