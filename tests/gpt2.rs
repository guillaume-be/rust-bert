use rust_bert::gpt2::{
    GPT2LMHeadModel, Gpt2Config, Gpt2ConfigResources, Gpt2MergesResources, Gpt2ModelResources,
    Gpt2VocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::conversation::{
    ConversationConfig, ConversationManager, ConversationModel,
};
use rust_bert::pipelines::generation_utils::{Cache, GenerateConfig, LMHeadModel};
use rust_bert::pipelines::text_generation::TextGenerationModel;
use rust_bert::resources::{RemoteResource, Resource};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, Device, Tensor};

#[test]
fn gpt2_lm_model() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let weights_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let merges_path = merges_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        false,
    )?;
    let config = Gpt2Config::from_file(config_path);
    let gpt2_model = GPT2LMHeadModel::new(&vs.root(), &config);
    vs.load(weights_path)?;

    //    Define input
    let input = ["One two three four"];
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
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let model_output = gpt2_model
        .forward_t(
            &Some(input_tensor),
            Cache::None,
            &None,
            &None,
            &None,
            &None,
            None,
            &None,
            false,
        )
        .unwrap();

    let next_word_id = model_output
        .lm_logits
        .get(0)
        .get(-1)
        .argmax(-1, true)
        .int64_value(&[0]);
    let next_word = tokenizer.decode(vec![next_word_id], true, true);

    assert_eq!(model_output.lm_logits.size(), vec!(1, 4, 50257));
    match model_output.cache {
        Cache::GPT2Cache(past) => {
            assert!(past.is_some());
            assert_eq!(past.as_ref().unwrap().len(), config.n_layer as usize);
            assert_eq!(
                past.as_ref().unwrap()[0].size(),
                vec!(2, 1, config.n_head, 4, 64)
            );
        }
        _ => panic!("Wrong cache returned for GPT2"),
    }
    assert!(
        (model_output.lm_logits.double_value(&[
            0,
            model_output.lm_logits.size()[1] - 1,
            next_word_id
        ]) - (-69.4948))
            .abs()
            < 1e-4
    );
    assert_eq!(next_word_id, 1936i64);
    assert_eq!(next_word, String::from(" five"));

    Ok(())
}

#[test]
fn gpt2_generation_greedy() -> anyhow::Result<()> {
    //    Resources definition
    let config_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let model_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

    //    Set-up masked LM model
    let generate_config = GenerateConfig {
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        max_length: 40,
        do_sample: false,
        num_beams: 1,
        temperature: 1.1,
        repetition_penalty: 1.1,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config, ModelType::GPT2)?;

    let input_context = "The cat";
    let output = model.generate(&[input_context], None);

    assert_eq!(output.len(), 1);
    assert_eq!(output[0], "The cat was found in a field near the town of Keflavik, about 30 miles (48 kilometers) south-east of Moscow.\n\n\n");

    Ok(())
}

#[test]
fn gpt2_generation_beam_search() -> anyhow::Result<()> {
    //    Resources definition
    let config_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let model_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

    //    Set-up masked LM model
    let generate_config = GenerateConfig {
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        max_length: 20,
        do_sample: false,
        num_beams: 5,
        temperature: 1.2,
        num_return_sequences: 3,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config, ModelType::GPT2)?;

    let input_context = "The dog";
    let output = model.generate(&[input_context], None);

    assert_eq!(output.len(), 3);
    assert_eq!(
        output[0],
        "The dog was found in the backyard of a home in the 6200 block of South Main Street."
    );
    assert_eq!(
        output[1],
        "The dog was found in the backyard of a home in the 6500 block of South Main Street."
    );
    assert_eq!(
        output[2],
        "The dog was found in the backyard of a home in the 6200 block of South Main Street,"
    );

    Ok(())
}

#[test]
fn gpt2_generation_beam_search_multiple_prompts_without_padding() -> anyhow::Result<()> {
    //    Resources definition
    let config_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let model_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

    //    Set-up masked LM model
    let generate_config = GenerateConfig {
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        max_length: 20,
        do_sample: false,
        num_beams: 5,
        temperature: 1.2,
        num_return_sequences: 3,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config, ModelType::GPT2)?;

    let input_context_1 = "The dog";
    let input_context_2 = "The cat";
    let output = model.generate(&[input_context_1, input_context_2], None);

    assert_eq!(output.len(), 6);
    assert_eq!(
        output[0],
        "The dog was found in the backyard of a home in the 6200 block of South Main Street."
    );
    assert_eq!(
        output[1],
        "The dog was found in the backyard of a home in the 6500 block of South Main Street."
    );
    assert_eq!(
        output[2],
        "The dog was found in the backyard of a home in the 6200 block of South Main Street,"
    );
    assert_eq!(
        output[3],
        "The cat-and-mouse game.\n\n\"I think it\'s going to be interesting to"
    );
    assert_eq!(
        output[4],
        "The cat-and-mouse game.\n\n\"I think it\'s going to be a very"
    );
    assert_eq!(
        output[5],
        "The cat-and-mouse game.\n\n\"I think it\'s going to be very interesting"
    );

    Ok(())
}

#[test]
fn gpt2_generation_beam_search_multiple_prompts_with_padding() -> anyhow::Result<()> {
    //    Resources definition
    let config_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let model_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

    //    Set-up masked LM model
    let generate_config = GenerateConfig {
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        max_length: 20,
        do_sample: false,
        num_beams: 5,
        temperature: 1.2,
        num_return_sequences: 3,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config, ModelType::GPT2)?;

    let input_context_1 = "The dog";
    let input_context_2 = "The cat was";
    let output = model.generate(&[input_context_1, input_context_2], None);

    assert_eq!(output.len(), 6);
    assert_eq!(
        output[0],
        "The dog was found dead on the side of the road in the middle of the night.\n"
    );
    assert_eq!(
        output[1],
        "The dog was found dead on the side of the road in the middle of the night on Sunday"
    );
    assert_eq!(
        output[2],
        "The dog was found dead on the side of the road in the middle of the night on Saturday"
    );
    assert_eq!(
        output[3],
        "The cat was taken to a local hospital, where it was treated and released.\n\nPolice said"
    );
    assert_eq!(
        output[4],
        "The cat was taken to a local hospital, where it was treated and released.\n\n\"It"
    );
    assert_eq!(
        output[5],
        "The cat was taken to a local hospital, where it was treated and released.\n\n\"We"
    );

    Ok(())
}

#[test]
#[cfg_attr(not(feature = "all-tests"), ignore)]
fn dialogpt_single_multi_turn_conversation() -> anyhow::Result<()> {
    //    Set-up conversation model
    let conversation_config = ConversationConfig {
        do_sample: false,
        device: Device::Cpu,
        ..Default::default()
    };
    let conversation_model = ConversationModel::new(conversation_config)?;

    // Set-up conversation manager and add a conversation
    let mut conversation_manager = ConversationManager::new();
    let conversation_id =
        conversation_manager.create("Going to the movies tonight - any suggestions?");

    // Turn 1
    let output = conversation_model.generate_responses(&mut conversation_manager);
    assert_eq!(output.len(), 1);
    assert_eq!(output.get(&conversation_id).unwrap(), &"The Big Lebowski");

    // Turn 2
    let _ = conversation_manager
        .get(&conversation_id)
        .unwrap()
        .add_user_input("Is it an action movie?");
    let output = conversation_model.generate_responses(&mut conversation_manager);
    assert_eq!(output.len(), 1);
    assert_eq!(output.get(&conversation_id).unwrap(), &"It\'s a comedy.");

    // Turn 3 (no new user input)
    let output = conversation_model.generate_responses(&mut conversation_manager);
    assert_eq!(output.len(), 0);

    Ok(())
}

#[test]
#[cfg_attr(not(feature = "all-tests"), ignore)]
fn dialogpt_multiple_multi_turn_conversation() -> anyhow::Result<()> {
    //    Set-up conversation model
    let conversation_config = ConversationConfig {
        do_sample: false,
        device: Device::Cpu,
        ..Default::default()
    };
    let conversation_model = ConversationModel::new(conversation_config)?;

    // Set-up conversation manager and add a conversation
    let mut conversation_manager = ConversationManager::new();
    let conversation_1_id =
        conversation_manager.create("Going to the movies tonight - any suggestions?");
    let conversation_2_id = conversation_manager.create("What's the last book you have read?");

    // Turn 1
    let output = conversation_model.generate_responses(&mut conversation_manager);
    assert_eq!(output.len(), 2);
    assert_eq!(output.get(&conversation_1_id).unwrap(), &"The Big Lebowski");
    assert_eq!(
        output.get(&conversation_2_id).unwrap(),
        &"The Last Question"
    );

    // Turn 2
    let _ = conversation_manager
        .get(&conversation_1_id)
        .unwrap()
        .add_user_input("Is it an action movie?");
    let output = conversation_model.generate_responses(&mut conversation_manager);
    assert_eq!(output.len(), 1);
    assert_eq!(output.get(&conversation_1_id).unwrap(), &"It\'s a comedy.");

    // Turn 3 (no new user input)
    let output = conversation_model.generate_responses(&mut conversation_manager);
    assert_eq!(output.len(), 0);

    Ok(())
}

#[test]
#[cfg_attr(not(feature = "all-tests"), ignore)]
fn dialogpt_multiple_multi_turn_conversation_with_truncation() -> anyhow::Result<()> {
    //    Set-up conversation model
    let conversation_config = ConversationConfig {
        max_length: 36,
        min_length_for_response: 24,
        do_sample: false,
        device: Device::Cpu,
        ..Default::default()
    };
    let conversation_model = ConversationModel::new(conversation_config)?;

    // Set-up conversation manager and add a conversation
    let mut conversation_manager = ConversationManager::new();
    let conversation_1_id =
        conversation_manager.create("Going to the movies tonight - any suggestions?");

    // Turn 1
    let output = conversation_model.generate_responses(&mut conversation_manager);
    assert_eq!(output.len(), 1);
    assert_eq!(output.get(&conversation_1_id).unwrap(), &"The Big Lebowski");

    // Turn 2
    let _ = conversation_manager
        .get(&conversation_1_id)
        .unwrap()
        .add_user_input("Is it an action movie?");
    let output = conversation_model.generate_responses(&mut conversation_manager);
    assert_eq!(output.len(), 1);
    assert_eq!(output.get(&conversation_1_id).unwrap(), &"It\'s a comedy.");

    // Turn 3 (no new user input)
    let output = conversation_model.generate_responses(&mut conversation_manager);
    assert_eq!(output.len(), 0);

    Ok(())
}

#[test]
#[cfg_attr(not(feature = "all-tests"), ignore)]
fn dialogpt_multiple_multi_turn_conversation_with_conversation_deletion() -> anyhow::Result<()> {
    //    Set-up conversation model
    let conversation_config = ConversationConfig {
        do_sample: false,
        device: Device::Cpu,
        ..Default::default()
    };
    let conversation_model = ConversationModel::new(conversation_config)?;

    // Set-up conversation manager and add a conversation
    let mut conversation_manager = ConversationManager::new();
    let conversation_1_id =
        conversation_manager.create("Going to the movies tonight - any suggestions?");
    let conversation_2_id = conversation_manager.create("What's the last book you have read?");

    // Turn 1
    let output = conversation_model.generate_responses(&mut conversation_manager);
    assert_eq!(output.len(), 2);
    assert_eq!(output.get(&conversation_1_id).unwrap(), &"The Big Lebowski");
    assert_eq!(
        output.get(&conversation_2_id).unwrap(),
        &"The Last Question"
    );

    // Turn 2
    let _ = conversation_manager.remove(&conversation_1_id);
    let _ = conversation_manager
        .get(&conversation_2_id)
        .unwrap()
        .add_user_input("Why do you recommend it?");
    let output = conversation_model.generate_responses(&mut conversation_manager);
    assert_eq!(output.len(), 1);
    assert_eq!(
        output.get(&conversation_2_id).unwrap(),
        &"It's a good book."
    );

    // Turn 3 (no new user input)
    let output = conversation_model.generate_responses(&mut conversation_manager);
    assert_eq!(output.len(), 0);

    Ok(())
}
