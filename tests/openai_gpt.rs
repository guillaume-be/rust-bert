use tch::{Device, nn, Tensor};
use rust_tokenizers::{TruncationStrategy, Tokenizer, OpenAiGptTokenizer};
use rust_bert::Config;
use rust_bert::pipelines::generation::{OpenAIGenerator, LanguageGenerator, GenerateConfig, LMHeadModel};
use rust_bert::gpt2::Gpt2Config;
use rust_bert::openai_gpt::{OpenAIGPTLMHeadModel, OpenAiGptConfigResources, OpenAiGptVocabResources, OpenAiGptMergesResources, OpenAiGptModelResources};
use rust_bert::resources::{RemoteResource, Resource, download_resource};

#[test]
fn openai_gpt_lm_model() -> failure::Fallible<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptConfigResources::GPT));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptVocabResources::GPT));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptMergesResources::GPT));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptModelResources::GPT));
    let config_path = download_resource(&config_resource)?;
    let vocab_path = download_resource(&vocab_resource)?;
    let merges_path = download_resource(&merges_resource)?;
    let weights_path = download_resource(&weights_resource)?;

//    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer = OpenAiGptTokenizer::from_file(vocab_path.to_str().unwrap(), merges_path.to_str().unwrap(), true);
    let config = Gpt2Config::from_file(config_path);
    let openai_gpt = OpenAIGPTLMHeadModel::new(&vs.root(), &config);
    vs.load(weights_path)?;

//    Define input
    let input = ["Wondering what the next word will"];
    let tokenized_input = tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
    let max_len = tokenized_input.iter().map(|input| input.token_ids.len()).max().unwrap();
    let tokenized_input = tokenized_input.
        iter().
        map(|input| input.token_ids.clone()).
        map(|mut input| {
            input.extend(vec![0; max_len - input.len()]);
            input
        }).
        map(|input|
            Tensor::of_slice(&(input))).
        collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

//    Forward pass
    let (output, _, _, _, _) = openai_gpt.forward_t(
        &Some(input_tensor),
        &None,
        &None,
        &None,
        &None,
        &None,
        None,
        &None,
        false).unwrap();

    let next_word_id = output.get(0).get(-1).argmax(-1, true).int64_value(&[0]);
    let next_word = tokenizer.decode(vec!(next_word_id), true, true);

    assert_eq!(output.size(), vec!(1, 6, 40478));
    assert!((output.double_value(&[0, output.size()[1] - 1, next_word_id]) - (9.1056)).abs() < 1e-4);
    assert_eq!(next_word_id, 580i64);
    assert_eq!(next_word, String::from("be"));

    Ok(())
}

#[test]
fn openai_gpt_generation_greedy() -> failure::Fallible<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptConfigResources::GPT));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptVocabResources::GPT));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptMergesResources::GPT));
    let model_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptModelResources::GPT));

//    Set-up masked LM model
    let generate_config = GenerateConfig {
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        max_length: 40,
        do_sample: false,
        num_beams: 1,
        top_p: 1.0,
        no_repeat_ngram_size: 1,
        temperature: 1.1,
        ..Default::default()
    };
    let mut model = OpenAIGenerator::new(generate_config)?;

    let input_context = "It was an intense machine dialogue. ";
    let output = model.generate(Some(vec!(input_context)), None);

    assert_eq!(output.len(), 1);
    assert_eq!(output[0], "it was an intense machine dialogue. \n \" i\'m sorry, but we have to go now! the police are on their way and they\'re going after you - or at least that\'s what my");

    Ok(())
}

#[test]
fn openai_gpt_generation_beam_search() -> failure::Fallible<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptConfigResources::GPT));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptVocabResources::GPT));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptMergesResources::GPT));
    let model_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptModelResources::GPT));

//    Set-up masked LM model
    let generate_config = GenerateConfig {
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        max_length: 20,
        do_sample: false,
        num_beams: 5,
        temperature: 2.0,
        num_return_sequences: 3,
        ..Default::default()
    };
    let mut model = OpenAIGenerator::new(generate_config)?;

    let input_context = "The dog is";
    let output = model.generate(Some(vec!(input_context)), None);

    assert_eq!(output.len(), 3);
    assert_eq!(output[0], "the dog isn\'t going anywhere. i\'m going to take care of him. i \'ll be right");
    assert_eq!(output[1], "the dog isn\'t going anywhere. i\'m going to take care of him. i \'ll be back");
    assert_eq!(output[2], "the dog isn\'t going anywhere. i\'m going to take care of him. \" \n \" i");

    Ok(())
}

#[test]
fn openai_gpt_generation_beam_search_multiple_prompts_without_padding() -> failure::Fallible<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptConfigResources::GPT));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptVocabResources::GPT));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptMergesResources::GPT));
    let model_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptModelResources::GPT));

//    Set-up masked LM model
    let generate_config = GenerateConfig {
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        max_length: 20,
        do_sample: false,
        num_beams: 5,
        temperature: 2.0,
        num_return_sequences: 3,
        ..Default::default()
    };
    let mut model = OpenAIGenerator::new(generate_config)?;

    let input_context_1 = "The dog is";
    let input_context_2 = "The cat";
    let output = model.generate(Some(vec!(input_context_1, input_context_2)), None);

    assert_eq!(output.len(), 6);

//    Unpadded sequence (generation for `The dog is`) is identical to the
    assert_eq!(output[0], "the dog isn\'t going anywhere. i\'m going to take care of him. i \'ll be right");
    assert_eq!(output[1], "the dog isn\'t going anywhere. i\'m going to take care of him. i \'ll be back");
    assert_eq!(output[2], "the dog isn\'t going anywhere. i\'m going to take care of him. \" \n \" i");

    assert_eq!(output[3], "the cat. \" \n \" i don\'t know what you\'re talking about. i don\'t");
    assert_eq!(output[4], "the cat. \" \n \" i don\'t know what you\'re talking about. i\'m not");
    assert_eq!(output[5], "the cat. \" \n \" i don\'t know what you\'re talking about. i do know");

    Ok(())
}

#[test]
fn openai_gpt_generation_beam_search_multiple_prompts_with_padding() -> failure::Fallible<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptConfigResources::GPT));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptVocabResources::GPT));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptMergesResources::GPT));
    let model_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptModelResources::GPT));

//    Set-up masked LM model
    let generate_config = GenerateConfig {
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        max_length: 20,
        do_sample: false,
        num_beams: 5,
        temperature: 2.0,
        num_return_sequences: 3,
        ..Default::default()
    };
    let mut model = OpenAIGenerator::new(generate_config)?;

    let input_context_1 = "The dog is";
    let input_context_2 = "The cat was in";
    let output = model.generate(Some(vec!(input_context_1, input_context_2)), None);

    assert_eq!(output.len(), 6);
//    Left padding impacts the generated sentences output
    assert_eq!(output[0], "the dog is a dog. \" \n \" i don\'t know what you\'re talking about.");
    assert_eq!(output[1], "the dog is a dog. \" \n \" i don\'t know what you\'re talking about,");
    assert_eq!(output[2], "the dog is a dog. \" \n \" i don\'t know what you\'re talking about!");
    assert_eq!(output[3], "the cat was in the room with them. \n \" what\'s going on? \" i asked.");
    assert_eq!(output[4], "the cat was in the room with them. \n \" what\'s going on? \" she asked.");
    assert_eq!(output[5], "the cat was in the room with them. \n \" what\'s going on? why are you all");

    Ok(())
}
