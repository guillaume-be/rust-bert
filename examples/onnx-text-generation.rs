use rust_bert::pipelines::common::{ModelResource, ModelType, ONNXModelResources};
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::resources::RemoteResource;

fn main() -> anyhow::Result<()> {
    let text_generation_model = TextGenerationModel::new(TextGenerationConfig {
        model_type: ModelType::GPT2,
        model_resource: ModelResource::ONNX(ONNXModelResources {
            encoder_resource: None,
            decoder_resource: Some(Box::new(RemoteResource::new(
                "https://huggingface.co/optimum/gpt2/resolve/main/decoder_model.onnx",
                "onnx-gpt2",
            ))),
            decoder_with_past_resource: Some(Box::new(RemoteResource::new(
                "https://huggingface.co/optimum/gpt2/resolve/main/decoder_with_past_model.onnx",
                "onnx-gpt2",
            ))),
        }),
        config_resource: Box::new(RemoteResource::new(
            "https://huggingface.co/optimum/gpt2/resolve/main/config.json",
            "onnx-gpt2",
        )),
        vocab_resource: Box::new(RemoteResource::new(
            "https://huggingface.co/gpt2/resolve/main/vocab.json",
            "onnx-gpt2",
        )),
        merges_resource: Some(Box::new(RemoteResource::new(
            "https://huggingface.co/gpt2/resolve/main/merges.txt",
            "onnx-gpt2",
        ))),
        max_length: Some(30),
        do_sample: false,
        num_beams: 1,
        temperature: 1.0,
        num_return_sequences: 1,
        ..Default::default()
    })?;
    let prompts = ["It was a very nice and sunny"];
    let output = text_generation_model.generate(&prompts, None);
    println!("{:?}", output);
    Ok(())
}
