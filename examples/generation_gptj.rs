use std::path::PathBuf;

use rust_bert::gpt_j::{GptJConfigResources, GptJMergesResources, GptJVocabResources};
use rust_bert::pipelines::common::{ModelResource, ModelType};
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::resources::{LocalResource, RemoteResource};
use tch::Device;

/// Equivalent Python code:
///
/// ```python
/// import torch
/// from transformers import AutoTokenizer, GPTJForCausalLM
///
/// device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
///
/// model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16).to(device)
///
/// tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", padding_side="left")
/// tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
///
/// prompts = ["It was a very nice and sunny", "It was a gloom winter night, and"]
/// inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
///
/// with torch.no_grad():
///     gen_tokens = model.generate(
///         **inputs,
///         min_length=0,
///         max_length=32,
///         do_sample=False,
///         early_stopping=True,
///         num_beams=1,
///         num_return_sequences=1
///     )
///
/// gen_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
/// ````
///
/// To run this test you need to download `pytorch_model.bin` from [EleutherAI GPT-J 6B
/// (float16)][gpt-j-6B-float16] and then convert its weights with:
///
/// ```
/// python utils/convert_model.py resources/gpt-j-6B-float16/pytorch_model.bin
/// ```
///
/// [gpt-j-6B-float16]: https://huggingface.co/EleutherAI/gpt-j-6B/tree/float16
fn main() -> anyhow::Result<()> {
    // Resources paths

    let config_resource = Box::new(RemoteResource::from_pretrained(
        GptJConfigResources::GPT_J_6B_FLOAT16,
    ));

    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        GptJVocabResources::GPT_J_6B_FLOAT16,
    ));

    let merges_resource = Box::new(RemoteResource::from_pretrained(
        GptJMergesResources::GPT_J_6B_FLOAT16,
    ));

    let model_resource = Box::new(LocalResource::from(PathBuf::from(
        "resources/gpt-j-6B-float16/rust_model.ot",
    )));

    // Set-up model

    let generation_config = TextGenerationConfig {
        model_type: ModelType::GPTJ,
        model_resource: ModelResource::Torch(model_resource),
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        min_length: 10,
        max_length: Some(32),
        do_sample: false,
        early_stopping: true,
        num_beams: 1,
        num_return_sequences: 1,
        device: Device::cuda_if_available(),
        ..Default::default()
    };

    let model = TextGenerationModel::new(generation_config)?;

    // Generate text

    let prompts = [
        "It was a very nice and sunny",
        "It was a gloom winter night, and",
    ];
    let output = model.generate(&prompts, None);

    assert_eq!(output.len(), 2);
    assert_eq!(output[0], "It was a very nice and sunny day, and I was sitting in the garden of my house, enjoying the sun and the fresh air. I was thinking");
    assert_eq!(output[1], "It was a gloom winter night, and the wind was howling. The snow was falling, and the temperature was dropping. The snow was coming down so hard");

    Ok(())
}
