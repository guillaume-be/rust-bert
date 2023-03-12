use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ndarray::Array2;
use ort::{
    tensor::{FromArray, InputTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, Session, SessionBuilder,
};

struct NameMapping<'a> {
    decoder_input_names: HashMap<&'a str, usize>,
    decoder_output_names: HashMap<&'a str, usize>,
    decoder_key_value_input_names: HashMap<&'a str, usize>,
    decoder_key_value_output_names: HashMap<&'a str, usize>,
}

fn get_input_output_mapping(session: &Session) -> NameMapping {
    let decoder_input_names = session
        .inputs
        .iter()
        .enumerate()
        .map(|(pos, input)| (input.name.as_str(), pos))
        .collect::<HashMap<&str, usize>>();

    let decoder_output_names = session
        .outputs
        .iter()
        .enumerate()
        .map(|(pos, output)| (output.name.as_str(), pos))
        .collect::<HashMap<&str, usize>>();

    let decoder_key_value_input_names = decoder_input_names
        .iter()
        .filter(|(name, _)| name.contains(".key") | name.contains(".value"))
        .map(|(name, pos)| (*name, *pos))
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
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .with_model_from_file(PathBuf::from(
            "E:/Coding/distilgpt2-onnx/decoder_model.onnx",
        ))?;

    // ToDo: align the input array using the mapping
    let decoder_name_mapping = get_input_output_mapping(&decoder_session);

    let input_ids = Array2::from_shape_vec((1, 3), vec![8888i64, 318i64, 257i64]).unwrap();
    let attention_mask = Array2::from_shape_vec((1, 3), vec![1i64, 1i64, 1i64]).unwrap();

    let outputs = decoder_session.run([
        InputTensor::from_array(input_ids.into_dyn()),
        InputTensor::from_array(attention_mask.into_dyn()),
    ])?;

    // ToDo: extract the features using the mapping
    println!("{:?}", outputs);

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
