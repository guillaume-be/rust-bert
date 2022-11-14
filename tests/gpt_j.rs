use rust_bert::gpt_j::{
    GptJConfig, GptJConfigResources, GptJLMHeadModel, GptJMergesResources, GptJModelResources,
    GptJVocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::generation_utils::{Cache, LMHeadModel};
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, MultiThreadedTokenizer, Tokenizer};
use rust_tokenizers::vocab::Gpt2Vocab;
use tch::{nn, Device, Tensor};

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
/// prompts = [
///     "It was a very nice and sunny",
///     "It was a gloom winter night, and"
/// ]
///
/// inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
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
#[test]
#[cfg_attr(not(feature = "all-tests"), ignore)]
fn gpt_j_generation() -> anyhow::Result<()> {
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

    let model_resource = Box::new(RemoteResource::from_pretrained(
        GptJModelResources::GPT_J_6B_FLOAT16,
    ));

    // Set-up model

    let generation_config = TextGenerationConfig {
        model_type: ModelType::GPTJ,
        model_resource,
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

/// Equivalent Python code:
///
/// ```python
/// import torch
/// from transformers import AutoTokenizer, GPTJForCausalLM
///
/// device = "cpu
///
/// model = GPTJForCausalLM.from_pretrained("anton-l/gpt-j-tiny-random").to(device)
///
/// tokenizer = AutoTokenizer.from_pretrained("anton-l/gpt-j-tiny-random", padding_side="left")
/// tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
///
/// prompts = ["It was a very nice and sunny", "It was a gloom winter night, and"]
/// inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
///
/// with torch.no_grad():
///     model.forward(**inputs).logits
/// ```
#[test]
fn gpt_j_correctness() -> anyhow::Result<()> {
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

    let device = Device::Cpu;

    // Set-up tokenizer

    let vocab_path = vocab_resource.get_local_path()?;
    let merges_path = merges_resource.get_local_path()?;
    let lower_case = false;
    let tokenizer = Gpt2Tokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        lower_case,
    )?;

    // Set-up model

    let mut vs = nn::VarStore::new(device);
    let config_path = config_resource.get_local_path()?;
    let weights_path = model_resource.get_local_path()?;
    let mut config = GptJConfig::from_file(config_path);
    config.use_float16 = false;
    let model = GptJLMHeadModel::new(&vs.root(), &config);
    vs.load(weights_path)?;

    // Tokenize prompts

    let prompts = [
        "It was a very nice and sunny",
        "It was a gloom winter night, and",
    ];

    let &pad_token = MultiThreadedTokenizer::vocab(&tokenizer)
        .special_values
        .get(Gpt2Vocab::eos_value())
        .unwrap_or(&2);

    let tokens = MultiThreadedTokenizer::tokenize_list(&tokenizer, &prompts);
    let token_ids = tokens
        .into_iter()
        .map(|prompt_tokens| tokenizer.convert_tokens_to_ids(&prompt_tokens))
        .collect::<Vec<Vec<i64>>>();

    let max_len = token_ids.iter().map(|input| input.len()).max().unwrap();

    let token_ids = token_ids
        .into_iter()
        .map(|input| {
            let mut temp = vec![pad_token; max_len - input.len()];
            temp.extend(input);
            temp
        })
        .map(|tokens| Tensor::of_slice(&tokens).to(device))
        .collect::<Vec<Tensor>>();

    let input_tensor = Tensor::stack(&token_ids, 0);

    // Run model inference

    let logits = tch::no_grad(|| {
        model.forward_t(
            Some(&input_tensor),
            Cache::None,
            None,
            None,
            None,
            None,
            None,
            None,
            false,
        )
    })?
    .lm_logits;

    assert!((logits.double_value(&[0, 0, 0]) - -0.1117).abs() < 1e-4);
    assert!((logits.double_value(&[0, 0, 1]) - 0.0562).abs() < 1e-4);
    assert!((logits.double_value(&[0, 0, 2]) - 0.1275).abs() < 1e-4);
    assert!((logits.double_value(&[0, 0, 50397]) - -0.1872).abs() < 1e-4);
    assert!((logits.double_value(&[0, 0, 50398]) - -0.1110).abs() < 1e-4);
    assert!((logits.double_value(&[0, 0, 50399]) - -0.3047).abs() < 1e-4);

    assert!((logits.double_value(&[1, 0, 0]) - -0.0647).abs() < 1e-4);
    assert!((logits.double_value(&[1, 0, 1]) - 0.0105).abs() < 1e-4);
    assert!((logits.double_value(&[1, 0, 2]) - -0.3448).abs() < 1e-4);
    assert!((logits.double_value(&[1, 0, 50397]) - -0.0445).abs() < 1e-4);
    assert!((logits.double_value(&[1, 0, 50398]) - 0.0639).abs() < 1e-4);
    assert!((logits.double_value(&[1, 0, 50399]) - -0.1167).abs() < 1e-4);

    Ok(())
}
