use std::collections::HashMap;
use std::convert::TryInto;
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::Array2;
use ort::{
    tensor::{FromArray, InputTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, SessionBuilder,
};
use rust_bert::pipelines::generation_utils::{Cache, LMModelOutput};
use rust_bert::pipelines::onnx::conversion::{ort_tensor_to_tch, ONNXLayerCache};
use rust_bert::pipelines::onnx::decoder::get_input_output_mapping;
use rust_bert::RustBertError;

fn main() -> anyhow::Result<()> {
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
            InputTensor::from_array(input_dict.remove(input_name.as_str()).unwrap().into_dyn())
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
            if let Some(array) = input_dict.remove(input_name.as_str()) {
                Ok(InputTensor::from_array(array.into_dyn()))
            } else {
                match lm_output.cache {
                    Cache::ONNXCache(ref cache) => {
                        let value: ndarray::ArrayD<f32> = cache
                            .values
                            .get(&input_name.replace("past_key_values", "present"))
                            .unwrap()
                            .try_into()
                            .unwrap();
                        Ok(InputTensor::from_array(value.into_dyn()))
                    }
                    _ => {
                        return Err(RustBertError::ValueError(
                            "Cache not compatible with GPT-J Model".into(),
                        ));
                    }
                }
            }
        })
        .collect::<Result<Vec<_>, RustBertError>>()?;

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
