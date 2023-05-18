use rust_bert::gpt_neo::{
    GptNeoConfig, GptNeoConfigResources, GptNeoForCausalLM, GptNeoMergesResources,
    GptNeoModelResources, GptNeoVocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, Device, Tensor};

#[test]
fn gpt_neo_lm() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoConfigResources::GPT_NEO_125M,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoVocabResources::GPT_NEO_125M,
    ));
    let merges_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoMergesResources::GPT_NEO_125M,
    ));
    let weights_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoModelResources::GPT_NEO_125M,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let merges_path = merges_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        false,
    )?;
    let mut config = GptNeoConfig::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let gpt_neo_model = GptNeoForCausalLM::new(vs.root(), &config)?;
    vs.load(weights_path)?;

    //    Define input
    let input = ["It was a sunny"];
    let tokenized_input = tokenizer.encode_list(&input, 128, &TruncationStrategy::LongestFirst, 0);
    let max_len = tokenized_input
        .iter()
        .map(|input| input.token_ids.len())
        .max()
        .unwrap();
    let tokenized_input = tokenized_input
        .iter()
        .map(|input| input.token_ids.clone())
        .map(|mut input| {
            input.extend(vec![0; max_len - input.len()]);
            input
        })
        .map(|input| Tensor::from_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let model_output =
        gpt_neo_model.forward_t(Some(&input_tensor), None, None, None, None, None, false)?;

    let next_word_id = model_output
        .lm_logits
        .get(0)
        .get(-1)
        .argmax(-1, true)
        .int64_value(&[0]);
    let next_word = tokenizer.decode(&[next_word_id], true, true);
    let next_score = model_output
        .lm_logits
        .get(0)
        .get(-1)
        .double_value(&[next_word_id]);

    // Output
    assert_eq!(model_output.lm_logits.size(), vec!(1, 4, 50257));
    assert_eq!(next_word_id, 1110_i64);
    assert!((next_score - (-0.0279)).abs() < 1e-4);
    assert_eq!(next_word, String::from(" day"));

    // Attentions & hidden states
    assert!(model_output.all_attentions.is_some());
    assert_eq!(model_output.all_attentions.as_ref().unwrap().len(), 12);
    assert_eq!(
        model_output.all_attentions.as_ref().unwrap()[0].size(),
        vec![1, 12, 4, 4]
    );
    assert_eq!(
        model_output.all_attentions.as_ref().unwrap()[1].size(),
        vec![1, 12, 4, 4]
    );

    assert!(model_output.all_hidden_states.is_some());
    assert_eq!(model_output.all_hidden_states.as_ref().unwrap().len(), 12);
    assert_eq!(
        model_output.all_hidden_states.as_ref().unwrap()[0].size(),
        vec![1, 4, 768]
    );
    Ok(())
}

#[test]
fn test_generation_gpt_neo() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoConfigResources::GPT_NEO_125M,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoVocabResources::GPT_NEO_125M,
    ));
    let merges_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoMergesResources::GPT_NEO_125M,
    ));
    let model_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoModelResources::GPT_NEO_125M,
    ));

    //    Set-up model
    let generation_config = TextGenerationConfig {
        model_type: ModelType::GPTNeo,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        min_length: 10,
        max_length: Some(32),
        do_sample: false,
        early_stopping: true,
        num_beams: 4,
        num_return_sequences: 1,
        device: Device::Cpu,
        ..Default::default()
    };

    let model = TextGenerationModel::new(generation_config)?;

    let input_context_1 = "It was a very nice and sunny";
    let input_context_2 = "It was a gloom winter night, and";
    let output = model.generate(&[input_context_1, input_context_2], None);

    assert_eq!(output.len(), 2);
    assert_eq!(output[0], "It was a very nice and sunny day. The sun was shining through the clouds, and the sky was clear. The wind was blowing through the trees,");
    assert_eq!(output[1], "It was a gloom winter night, and the sky was dark and cold, and the wind was blowing thick and heavy.\n\n\"What\'s the matter?\"");

    Ok(())
}
