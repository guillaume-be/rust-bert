use rust_bert::openai_gpt::{
    OpenAIGPTLMHeadModel, OpenAiGptConfig, OpenAiGptConfigResources, OpenAiGptMergesResources,
    OpenAiGptModelResources, OpenAiGptVocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::generation_utils::Cache;
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{OpenAiGptTokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, Device, Tensor};

#[test]
fn openai_gpt_lm_model() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptConfigResources::GPT,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptVocabResources::GPT,
    ));
    let merges_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptMergesResources::GPT,
    ));
    let weights_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptModelResources::GPT,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let merges_path = merges_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer = OpenAiGptTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        true,
    )?;
    let config = OpenAiGptConfig::from_file(config_path);
    let openai_gpt = OpenAIGPTLMHeadModel::new(vs.root(), &config);
    vs.load(weights_path)?;

    //    Define input
    let input = ["Wondering what the next word will"];
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
    let model_output = openai_gpt
        .forward_t(
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
        .unwrap();

    let next_word_id = model_output
        .lm_logits
        .get(0)
        .get(-1)
        .argmax(-1, true)
        .int64_value(&[0]);
    let next_word = tokenizer.decode(&[next_word_id], true, true);

    assert_eq!(model_output.lm_logits.size(), vec!(1, 6, 40478));
    assert!(
        (model_output.lm_logits.double_value(&[
            0,
            model_output.lm_logits.size()[1] - 1,
            next_word_id
        ]) - (9.1056))
            .abs()
            < 1e-4
    );
    assert_eq!(next_word_id, 580i64);
    assert_eq!(next_word, String::from("be"));

    Ok(())
}

#[test]
fn openai_gpt_generation_greedy() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptConfigResources::GPT,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptVocabResources::GPT,
    ));
    let merges_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptMergesResources::GPT,
    ));
    let model_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptModelResources::GPT,
    ));

    //    Set-up model
    let generate_config = TextGenerationConfig {
        model_type: ModelType::OpenAiGpt,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        max_length: Some(40),
        do_sample: false,
        num_beams: 1,
        top_p: 1.0,
        no_repeat_ngram_size: 1,
        temperature: 1.1,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config)?;

    let input_context = "It was an intense machine dialogue. ";
    let output = model.generate(&[input_context], None);

    assert_eq!(output.len(), 1);
    assert_eq!(output[0], "it was an intense machine dialogue. \n \" i\'m sorry, but we have to go now! the police are on their way and they\'re going after you - or at least that\'s what my");

    Ok(())
}

#[test]
fn openai_gpt_generation_beam_search() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptConfigResources::GPT,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptVocabResources::GPT,
    ));
    let merges_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptMergesResources::GPT,
    ));
    let model_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptModelResources::GPT,
    ));

    //    Set-up model
    let generate_config = TextGenerationConfig {
        model_type: ModelType::OpenAiGpt,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        max_length: Some(20),
        do_sample: false,
        early_stopping: true,
        num_beams: 5,
        temperature: 1.0,
        num_return_sequences: 3,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config)?;

    let input_context = "The dog is";
    let output = model.generate(&[input_context], None);

    assert_eq!(output.len(), 3);
    assert_eq!(
        output[0],
        "the dog is a good dog. \" \n \" he's a good dog, \" i agreed."
    );
    assert_eq!(
        output[1],
        "the dog is a good dog. \" \n \" he\'s a good dog. \" \n \" he"
    );
    assert_eq!(
        output[2],
        "the dog is a good dog. \" \n \" he\'s a good dog. \" \n \" i"
    );

    Ok(())
}

#[test]
fn openai_gpt_generation_beam_search_multiple_prompts_without_padding() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptConfigResources::GPT,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptVocabResources::GPT,
    ));
    let merges_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptMergesResources::GPT,
    ));
    let model_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptModelResources::GPT,
    ));

    //    Set-up model
    let generate_config = TextGenerationConfig {
        model_type: ModelType::OpenAiGpt,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        max_length: Some(20),
        do_sample: false,
        early_stopping: true,
        num_beams: 5,
        temperature: 1.0,
        num_return_sequences: 3,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config)?;

    let input_context_1 = "The dog is";
    let input_context_2 = "The cat";
    let output = model.generate(&[input_context_1, input_context_2], None);

    assert_eq!(output.len(), 6);

    //    Un-padded sequence (generation for `The dog is`) is identical to the case with a unique input
    assert_eq!(
        output[0],
        "the dog is a good dog. \" \n \" he's a good dog, \" i agreed."
    );
    assert_eq!(
        output[1],
        "the dog is a good dog. \" \n \" he\'s a good dog. \" \n \" he"
    );
    assert_eq!(
        output[2],
        "the dog is a good dog. \" \n \" he\'s a good dog. \" \n \" i"
    );

    assert_eq!(
        output[3],
        "the cat. \" \n \" what? \" \n \" you heard me. \" \n \" i"
    );
    assert_eq!(
        output[4],
        "the cat. \" \n \" what? \" \n \" you heard me. \" \n \" no"
    );
    assert_eq!(
        output[5],
        "the cat. \" \n \" what? \" \n \" you heard me. \" \n \" oh"
    );

    Ok(())
}

#[test]
fn openai_gpt_generation_beam_search_multiple_prompts_with_padding() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptConfigResources::GPT,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptVocabResources::GPT,
    ));
    let merges_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptMergesResources::GPT,
    ));
    let model_resource = Box::new(RemoteResource::from_pretrained(
        OpenAiGptModelResources::GPT,
    ));

    //    Set-up model
    let generate_config = TextGenerationConfig {
        model_type: ModelType::OpenAiGpt,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        max_length: Some(20),
        do_sample: false,
        num_beams: 5,
        temperature: 2.0,
        num_return_sequences: 3,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config)?;

    let input_context_1 = "The dog is";
    let input_context_2 = "The cat was in";
    let output = model.generate(&[input_context_1, input_context_2], None);

    assert_eq!(output.len(), 6);
    //    Left padding impacts the generated sentences output
    assert_eq!(
        output[0],
        "the dog is a dog. \" \n \" i don\'t know what you\'re talking about."
    );
    assert_eq!(
        output[1],
        "the dog is a dog. \" \n \" i don\'t know what you\'re talking about,"
    );
    assert_eq!(
        output[2],
        "the dog is a dog. \" \n \" i don\'t know what you\'re talking about!"
    );
    assert_eq!(
        output[3],
        "the cat was in the room with them. \n \" what\'s going on? \" i asked."
    );
    assert_eq!(
        output[4],
        "the cat was in the room with them. \n \" what\'s going on? \" she asked."
    );
    assert_eq!(
        output[5],
        "the cat was in the room with them. \n \" what\'s going on? why are you all"
    );

    Ok(())
}
