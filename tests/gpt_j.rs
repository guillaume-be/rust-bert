use rust_bert::gpt_j::{
    GptJConfig, GptJConfigResources, GptJLMHeadModel, GptJMergesResources, GptJModelResources,
    GptJVocabResources,
};
use rust_bert::pipelines::generation_utils::Cache;
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer};
use rust_tokenizers::vocab::Vocab;
use tch::{nn, Device, Tensor};

/// Equivalent Python code:
///
/// ```python
/// import torch
/// from transformers import AutoTokenizer, GPTJForCausalLM
///
/// device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
///
/// model = GPTJForCausalLM.from_pretrained("anton-l/gpt-j-tiny-random").to(device)
/// if torch.cuda.is_available(): model = model.half()
///
/// tokenizer = AutoTokenizer.from_pretrained("anton-l/gpt-j-tiny-random", padding_side="left")
/// tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
///
/// prompts = ["It was a very nice and sunny", "It was a gloom winter night, and"]
/// inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
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

    let device = Device::cuda_if_available();

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
    config.use_float16 = matches!(device, Device::Cuda(_));
    let model = GptJLMHeadModel::new(vs.root(), &config);
    vs.load(weights_path)?;

    // Tokenize prompts

    let prompts = [
        "It was a very nice and sunny",
        "It was a gloom winter night, and",
    ];

    let pad_token = tokenizer.vocab().get_eos_value();
    let &pad_token = tokenizer
        .vocab()
        .special_values()
        .get(pad_token)
        .unwrap_or(&2);

    let tokens = Tokenizer::tokenize_list(&tokenizer, &prompts);
    let max_len = tokens.iter().map(|input| input.len()).max().unwrap_or(0);

    let token_ids = tokens
        .into_iter()
        .map(|prompt_tokens| {
            let token_ids = tokenizer.convert_tokens_to_ids(&prompt_tokens);
            let mut padded = vec![pad_token; max_len - token_ids.len()];
            padded.extend(token_ids);
            padded
        })
        .collect::<Vec<Vec<i64>>>();

    let token_masks = token_ids
        .iter()
        .map(|input| {
            Tensor::from_slice(
                &input
                    .iter()
                    .map(|&e| i64::from(e != pad_token))
                    .collect::<Vec<_>>(),
            )
            .to(device)
        })
        .collect::<Vec<_>>();

    let token_ids = token_ids
        .into_iter()
        .map(|tokens| Tensor::from_slice(&tokens).to(device))
        .collect::<Vec<Tensor>>();

    let input_tensor = Tensor::stack(&token_ids, 0);
    let attention_tensor = Tensor::stack(&token_masks, 0);

    // Run model inference

    let logits = tch::no_grad(|| {
        model.forward_t(
            Some(&input_tensor),
            Cache::None,
            // None,
            Some(&attention_tensor),
            None,
            None,
            None,
            None,
            None,
            false,
        )
    })?
    .lm_logits;

    if matches!(device, Device::Cpu) {
        assert!((logits.double_value(&[0, 0, 0]) - -0.8343).abs() < 1e-4);
        assert!((logits.double_value(&[0, 0, 1]) - 0.0203).abs() < 1e-4);
        assert!((logits.double_value(&[0, 0, 2]) - 0.4745).abs() < 1e-4);
        assert!((logits.double_value(&[0, 0, 50397]) - 0.2641).abs() < 1e-4);
        assert!((logits.double_value(&[0, 0, 50398]) - 0.1926).abs() < 1e-4);
        assert!((logits.double_value(&[0, 0, 50399]) - 0.0204).abs() < 1e-4);

        assert!((logits.double_value(&[1, 0, 0]) - -0.0647).abs() < 1e-4);
        assert!((logits.double_value(&[1, 0, 1]) - 0.0105).abs() < 1e-4);
        assert!((logits.double_value(&[1, 0, 2]) - -0.3448).abs() < 1e-4);
        assert!((logits.double_value(&[1, 0, 50397]) - -0.0445).abs() < 1e-4);
        assert!((logits.double_value(&[1, 0, 50398]) - 0.0639).abs() < 1e-4);
        assert!((logits.double_value(&[1, 0, 50399]) - -0.1167).abs() < 1e-4);
    } else {
        assert!((logits.double_value(&[0, 0, 0]) - -0.1110).abs() < 1e-2);
        assert!((logits.double_value(&[0, 0, 1]) - 0.0565).abs() < 1e-2);
        assert!((logits.double_value(&[0, 0, 2]) - 0.1273).abs() < 1e-2);
        assert!((logits.double_value(&[0, 0, 50397]) - -0.1879).abs() < 1e-2);
        assert!((logits.double_value(&[0, 0, 50398]) - -0.1114).abs() < 1e-2);
        assert!((logits.double_value(&[0, 0, 50399]) - -0.3042).abs() < 1e-2);

        assert!((logits.double_value(&[1, 0, 0]) - -0.0651).abs() < 1e-2);
        assert!((logits.double_value(&[1, 0, 1]) - 0.0107).abs() < 1e-2);
        assert!((logits.double_value(&[1, 0, 2]) - -0.3452).abs() < 1e-2);
        assert!((logits.double_value(&[1, 0, 50397]) - -0.0436).abs() < 1e-2);
        assert!((logits.double_value(&[1, 0, 50398]) - 0.0645).abs() < 1e-2);
        assert!((logits.double_value(&[1, 0, 50399]) - -0.1166).abs() < 1e-2);
    }

    Ok(())
}
