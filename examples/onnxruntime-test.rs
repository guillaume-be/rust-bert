use std::collections::HashMap;
use std::convert::TryInto;
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::Array2;
use ort::{
    tensor::{FromArray, InputTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, Session, SessionBuilder,
};
use rust_bert::pipelines::generation_utils::{Cache, LMModelOutput};
use rust_bert::pipelines::onnx::{ort_tensor_to_tch, ONNXLayerCache};

#[derive(Debug)]
struct NameMapping<'a> {
    decoder_input_names: Vec<&'a str>,
    decoder_output_names: HashMap<&'a str, usize>,
    decoder_key_value_input_names: Vec<&'a str>,
    decoder_key_value_output_names: HashMap<&'a str, usize>,
}

fn get_input_output_mapping(session: &Session) -> NameMapping {
    let decoder_input_names = session
        .inputs
        .iter()
        .map(|input| input.name.as_str())
        .collect::<Vec<&str>>();

    let decoder_output_names = session
        .outputs
        .iter()
        .enumerate()
        .map(|(pos, output)| (output.name.as_str(), pos))
        .collect::<HashMap<&str, usize>>();

    let decoder_key_value_input_names = decoder_input_names
        .iter()
        .filter(|name| name.contains(".key") | name.contains(".value"))
        .map(|name| *name)
        .collect::<Vec<&str>>();

    let decoder_key_value_output_names = decoder_output_names
        .iter()
        .filter(|(name, _)| name.contains(".key") | name.contains(".value"))
        .map(|(name, pos)| (*name, *pos))
        .collect::<HashMap<&str, usize>>();

    NameMapping {
        decoder_input_names,
        decoder_output_names,
        decoder_key_value_input_names,
        decoder_key_value_output_names,
    }
}

fn main() -> OrtResult<()> {
    tracing_subscriber::fmt::init();

    // Initial set-up, load ONNX sessions
    let environment = Arc::new(
        Environment::builder()
            .with_name("GPT2")
            .with_execution_providers([ExecutionProvider::cpu()])
            .build()?,
    );

    let decoder_session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_model_from_file(PathBuf::from(
            "E:/Coding/distilgpt2-onnx/decoder_model.onnx",
        ))?;
    let decoder_name_mapping = get_input_output_mapping(&decoder_session);

    let decoder_with_past_session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_model_from_file(PathBuf::from(
            "E:/Coding/distilgpt2-onnx/decoder_with_past_model.onnx",
        ))?;
    let decoder_with_past_name_mapping = get_input_output_mapping(&decoder_with_past_session);

    // Initial decoder forward pass (without past)

    let input_ids = Array2::from_shape_vec((1, 3), vec![8888i64, 318i64, 257i64]).unwrap();
    let attention_mask = Array2::from_shape_vec((1, 3), vec![1i64, 1i64, 1i64]).unwrap();

    let mut input_dict =
        HashMap::from([("input_ids", input_ids), ("attention_mask", attention_mask)]);

    let inputs = decoder_name_mapping
        .decoder_input_names
        .iter()
        .map(|input_name| {
            InputTensor::from_array(input_dict.remove(input_name).unwrap().into_dyn())
        })
        .collect::<Vec<_>>();

    let outputs = decoder_session.run(inputs)?;

    let lm_logits = &outputs[*decoder_name_mapping
        .decoder_output_names
        .get("logits")
        .unwrap()];

    let lm_logits = ort_tensor_to_tch(lm_logits)?;

    let cache = Cache::ONNXCache(ONNXLayerCache::from_ort_output(
        &outputs,
        &decoder_name_mapping.decoder_key_value_output_names,
    )?);

    let lm_output = LMModelOutput { lm_logits, cache };

    println!("{} - {:?}", lm_output.lm_logits, lm_output.cache);

    // Second decoder forward pass (without past)

    let input_ids = Array2::from_shape_vec((1, 1), vec![649i64]).unwrap();
    let attention_mask = Array2::from_shape_vec((1, 4), vec![1i64, 1i64, 1i64, 1i64]).unwrap();

    let mut input_dict =
        HashMap::from([("input_ids", input_ids), ("attention_mask", attention_mask)]);

    let inputs = decoder_with_past_name_mapping
        .decoder_input_names
        .iter()
        .map(|input_name| {
            if let Some(array) = input_dict.remove(*input_name) {
                InputTensor::from_array(array.into_dyn())
            } else {
                match lm_output.cache {
                    Cache::ONNXCache(ref cache) => {
                        let value: ndarray::ArrayD<f32> = cache
                            .values
                            .get(&input_name.replace("past_key_values", "present"))
                            .unwrap()
                            .try_into()
                            .unwrap();
                        InputTensor::from_array(value.into_dyn())
                    }
                    _ => {
                        unreachable!()
                    }
                }
            }
        })
        .collect::<Vec<_>>();

    let outputs = decoder_with_past_session.run(inputs)?;

    let lm_logits = &outputs[*decoder_with_past_name_mapping
        .decoder_output_names
        .get("logits")
        .unwrap()];

    let lm_logits = ort_tensor_to_tch(lm_logits)?;

    let cache = Cache::ONNXCache(ONNXLayerCache::from_ort_output(
        &outputs,
        &decoder_with_past_name_mapping.decoder_key_value_output_names,
    )?);

    let lm_output = LMModelOutput { lm_logits, cache };

    println!("{} - {:?}", lm_output.lm_logits, lm_output.cache);

    Ok(())
}
