use crate::pipelines::onnx::config::ONNXEnvironmentConfig;
use crate::RustBertError;
use ort::{Environment, Session};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

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
    decoder_name_mapping: DecoderNameMapping,
}

impl ONNXDecoder {
    pub fn new(
        model_file: PathBuf,
        environment: &Arc<Environment>,
        onnx_config: &ONNXEnvironmentConfig,
    ) -> Result<Self, RustBertError> {
        let session = onnx_config
            .get_session_builder(environment)?
            .with_model_from_file(model_file)?;
        let decoder_name_mapping = get_input_output_mapping(&session);
        Ok(Self {
            session,
            decoder_name_mapping,
        })
    }
}
