use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::{Array2, Dimension};
use ort::{
    tensor::{FromArray, InputTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, Session, SessionBuilder,
};
use rust_bert::pipelines::generation_utils::{Cache, LMModelOutput};
use rust_bert::pipelines::onnx::ONNXLayerCache;
use tch::Tensor;

#[derive(Debug)]
struct NameMapping<'a> {
    decoder_input_names: Vec<&'a str>,
    decoder_output_names: HashMap<&'a str, usize>,
    decoder_key_value_input_names: HashMap<&'a str, usize>,
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
        .enumerate()
        .filter(|(_, name)| name.contains(".key") | name.contains(".value"))
        .map(|(pos, name)| (*name, pos))
        .collect::<HashMap<&str, usize>>();

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

    let logits = outputs[*decoder_name_mapping
        .decoder_output_names
        .get("logits")
        .unwrap()]
    .try_extract::<f32>()?;

    let lm_logits = Tensor::of_slice(logits.view().as_slice().unwrap()).view(
        logits
            .view()
            .dim()
            .as_array_view()
            .iter()
            .map(|dim| *dim as i64)
            .collect::<Vec<_>>()
            .as_slice(),
    );

    let cache = Cache::ONNXCache(ONNXLayerCache::from_ort_output(
        &outputs,
        &decoder_name_mapping.decoder_key_value_output_names,
    ));

    let lm_output = LMModelOutput { lm_logits, cache };

    println!("{} - {:?}", lm_output.lm_logits, lm_output.cache);

    // ToDo: identify if the input and output types for the models are the same (use as reference type for ONNXCache)
    // ToDo: store the output in a Cache object (to have the logits and cache in the dict)
    // ToDo: skip output dict and directly get the logits and present stored in the cache
    // ToDo: extract the features using the mapping
    // println!("{:?}", output_dict);

    // for _ in 0..GEN_TOKENS {
    //     let n_tokens = &tokens.shape()[0];
    //     let array = tokens
    //         .clone()
    //         .insert_axis(Axis(0))
    //         .into_shape((1, 1, *n_tokens))
    //         .unwrap();
    //     let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> =
    //         session.run([InputTensor::from_array(array.into_dyn())])?;
    //     let generated_tokens: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();
    //     let generated_tokens = generated_tokens.view();
    //
    //     let probabilities = &mut generated_tokens
    //         .slice(s![0, 0, -1, ..])
    //         .insert_axis(Axis(0))
    //         .to_owned()
    //         .iter()
    //         .cloned()
    //         .enumerate()
    //         .collect::<Vec<_>>();
    //     probabilities
    //         .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));
    //
    //     let token = probabilities[rng.gen_range(0..=TOP_K)].0;
    //     *tokens = concatenate![Axis(0), *tokens, array![token.try_into().unwrap()]];
    //     let sentence = tokenizer
    //         .decode(tokens.iter().map(|i| *i as u32).collect::<Vec<_>>(), true)
    //         .unwrap();
    //     println!("{}", sentence);
    // }

    Ok(())
}
