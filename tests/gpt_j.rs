use rust_bert::gpt_j::{
    GptJConfigResources, GptJMergesResources, GptJModelResources, GptJVocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::resources::RemoteResource;
use tch::Device;

/// Equivalent Python code:
///
/// ```python
/// import torch
/// from transformers import AutoTokenizer, GPTJForCausalLM
///
/// device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
///
/// model = GPTJForCausalLM.from_pretrained("anton-l/gpt-j-tiny-random").to(device)
///
/// tokenizer = AutoTokenizer.from_pretrained("anton-l/gpt-j-tiny-random", padding_side="left")
/// tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
///
/// inputs = [
///     "It was a very nice and sunny",
///     "It was a gloom winter night, and"
/// ]
///
/// prompts = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
///
/// with torch.no_grad():
///     gen_tokens = model.generate(
///         **prompts,
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
#[test]
fn test_generation_gpt_j() -> anyhow::Result<()> {
    // Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        GptJConfigResources::GPT_J_TINY_RANDOM,
    ));

    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        GptJVocabResources::GPT_J_TINY_RANDOM,
    ));

    let merges_resource = Box::new(RemoteResource::from_pretrained(
        GptJMergesResources::GPT_J_TINY_RANDOM,
    ));

    let model_resource = Box::new(RemoteResource::from_pretrained(
        GptJModelResources::GPT_J_TINY_RANDOM,
    ));

    // Set-up model
    let generation_config = TextGenerationConfig {
        model_type: ModelType::GPTJ,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        min_length: 10,
        max_length: 32,
        do_sample: false,
        early_stopping: true,
        num_beams: 1,
        num_return_sequences: 1,
        device: Device::cuda_if_available(),
        ..Default::default()
    };

    let mut model = TextGenerationModel::new(generation_config)?;
    model.float();

    let input_context_1 = "It was a very nice and sunny";
    let input_context_2 = "It was a gloom winter night, and";
    let output = model.generate(&[input_context_1, input_context_2], None);

    assert_eq!(output.len(), 2);
    assert_eq!(output[0], "It was a very nice and sunnyUkraine Joh revelation Phill camelzanne Wonderfulhing nicotine journalistic departures flashyourage Ralph Wonders DOS498Mur 65 License SukESE motions millennial");
    assert_eq!(output[1], "It was a gloom winter night, andisco revelation Phill listened incl dropped Ducks License Suk Techniciansterdam civilizations Republicans encourinka measurementfalserimp Trying CNS fugitive Schw License Suk");

    Ok(())
}
