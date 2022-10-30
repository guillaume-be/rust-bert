use std::path::Path;
use tract_onnx::prelude::*;

fn main() -> anyhow::Result<()> {
    let encoder_path = Path::new("E:/Coding/t5-small/encoder_model.onnx");
    let decoder_path = Path::new("E:/Coding/t5-small/decoder_model.onnx");
    let decoder_with_past_path = Path::new("E:/Coding/t5-small/decoder_with_past_model.onnx");

    let encoder_model = onnx().model_for_path(encoder_path)?.into_runnable()?;

    let input_ids: Vec<i64> = vec![8774, 48, 19, 3, 9, 182, 307, 1499, 12, 36, 15459, 5, 1];
    let attention: Vec<i64> = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

    let input_ids =
        tract_ndarray::Array2::from_shape_vec((1, input_ids.len()), input_ids.clone())?.into();
    let attention_mask =
        tract_ndarray::Array2::from_shape_vec((1, attention.len()), attention)?.into();

    let model_inputs = tvec!(input_ids, attention_mask);
    let result = encoder_model.run(model_inputs)?;

    println!("{:?}", result);

    Ok(())
}
