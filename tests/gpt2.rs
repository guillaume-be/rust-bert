use rust_bert::gpt2::{
    GPT2Generator, GPT2LMHeadModel, Gpt2Config, Gpt2ConfigResources, Gpt2MergesResources,
    Gpt2ModelResources, Gpt2VocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::conversation::{
    ConversationConfig, ConversationManager, ConversationModel,
};
use rust_bert::pipelines::generation_utils::{
    Cache, GenerateConfig, GenerateOptions, LanguageGenerator,
};
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, Device, Tensor};

#[test]
fn gpt2_lm_model() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2);
    let vocab_resource = RemoteResource::from_pretrained(Gpt2VocabResources::GPT2);
    let merges_resource = RemoteResource::from_pretrained(Gpt2MergesResources::GPT2);
    let weights_resource = RemoteResource::from_pretrained(Gpt2ModelResources::GPT2);
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let merges_path = merges_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        false,
    )?;
    let config = Gpt2Config::from_file(config_path);
    let gpt2_model = GPT2LMHeadModel::new(vs.root(), &config);
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
        .map(|input| Tensor::from_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let model_output = gpt2_model
        .forward_t(Some(&input_tensor), None, None, None, None, None, false)
        .unwrap();

    let next_word_id = model_output
        .lm_logits
        .get(0)
        .get(-1)
        .argmax(-1, true)
        .int64_value(&[0]);
    let next_word = tokenizer.decode(&[next_word_id], true, true);

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
    let config_resource = Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource = Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let model_resource = Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

    let generate_config = TextGenerationConfig {
        model_type: ModelType::GPT2,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        max_length: Some(40),
        do_sample: false,
        num_beams: 1,
        temperature: 1.1,
        repetition_penalty: 1.1,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config)?;

    let input_context = "The cat";
    let output = model.generate(&[input_context], None);

    assert_eq!(output.len(), 1);
    assert_eq!(output[0], "The cat was found in a field near the town of Keflavik, about 30 miles (48 kilometers) south-east of Moscow.\n\n\n");

    Ok(())
}

#[test]
fn gpt2_generation_beam_search() -> anyhow::Result<()> {
    //    Resources definition
    let config_resource = Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource = Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let model_resource = Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

    let generate_config = TextGenerationConfig {
        model_type: ModelType::GPT2,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        max_length: Some(20),
        do_sample: false,
        num_beams: 5,
        temperature: 1.2,
        device: Device::Cpu,
        num_return_sequences: 3,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config)?;

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
    let config_resource = Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource = Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let model_resource = Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

    let generate_config = TextGenerationConfig {
        model_type: ModelType::GPT2,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        max_length: Some(20),
        do_sample: false,
        num_beams: 5,
        temperature: 1.2,
        num_return_sequences: 3,
        device: Device::Cpu,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config)?;

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
    let config_resource = Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource = Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let model_resource = Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

    let generate_config = TextGenerationConfig {
        model_type: ModelType::GPT2,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        max_length: Some(20),
        do_sample: false,
        num_beams: 5,
        temperature: 1.2,
        num_return_sequences: 3,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config)?;

    let input_context_1 = "The dog";
    let input_context_2 = "The cat was";
    let output = model.generate(&[input_context_1, input_context_2], None);

    assert_eq!(output.len(), 6);
    assert_eq!(
        output[0],
        "The dog was found in the backyard of a home in the 6200 block of South Main Street"
    );
    assert_eq!(
        output[1],
        "The dog was found in the backyard of a home in the 6500 block of South Main Street"
    );
    assert_eq!(
        output[2],
        "The dog was found in the backyard of a home in the 6200 block of North Main Street"
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
fn gpt2_diverse_beam_search_multiple_prompts_with_padding() -> anyhow::Result<()> {
    //    Resources definition
    let config_resource = Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource = Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let model_resource = Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

    let generate_config = TextGenerationConfig {
        model_type: ModelType::GPT2,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        min_length: 10,
        max_length: Some(20),
        do_sample: false,
        num_beams: 6,
        num_beam_groups: Some(3),
        diversity_penalty: Some(5.5),
        num_return_sequences: 3,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config)?;

    let input_context_1 = "It was a nice and";
    let input_context_2 = "Language models can generate";
    let output = model.generate(&[input_context_1, input_context_2], None);

    assert_eq!(output.len(), 6);
    assert_eq!(
        output[0],
        "It was a nice and peaceful evening for me,\" he said.\n\n\"It was a good"
    );
    assert_eq!(
        output[1],
        "It was a nice and peaceful evening for me,\" he said.\n\n\"It was a nice"
    );
    assert_eq!(
        output[2],
        "It was a nice and warm day, and I\'m glad I did.\n\n\"I'm"
    );
    assert_eq!(
        output[3],
        "Language models can generate more complex models, but they are not the only way to do so."
    );
    assert_eq!(
        output[4],
        "Language models can generate more complex models, but they are not the only way to do this."
    );
    assert_eq!(
        output[5],
        "Language models can generate a lot of data, but they're not the only way to do it"
    );

    Ok(())
}

#[test]
fn gpt2_prefix_allowed_token_greedy() -> anyhow::Result<()> {
    //    Resources definition
    let config_resource = Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource = Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let model_resource = Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

    fn force_one_paragraph(_batch_id: i64, previous_token_ids: &Tensor) -> Vec<i64> {
        let paragraph_tokens = [198, 628];

        for paragraph_token in paragraph_tokens.iter() {
            if previous_token_ids
                .iter::<i64>()
                .unwrap()
                .any(|x| x == *paragraph_token)
            {
                return vec![50256];
            }
        }
        (0..50255).collect()
    }

    let generate_config = GenerateConfig {
        max_length: Some(56),
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        do_sample: false,
        num_beams: 1,
        device: Device::Cpu,
        ..Default::default()
    };
    let model = GPT2Generator::new(generate_config)?;

    let input_context_1 = "Rust is a";
    let input_context_2 = "There was a ";

    let generate_options = GenerateOptions {
        prefix_allowed_tokens_fn: Some(&force_one_paragraph),
        output_scores: true,
        ..Default::default()
    };

    let output = model.generate(
        Some(&[input_context_1, input_context_2]),
        Some(generate_options),
    );

    assert_eq!(output.len(), 2);
    assert_eq!(
        output[0].text,
        "Rust is a very simple and powerful library for building and running web applications. It is a simple, fast, and lightweight library that can be used to build web applications in a number of different ways.\n"
    );
    assert!((output[0].score.unwrap() - (-1.4666)).abs() < 1e-4);
    assert_eq!(
        output[1].text,
        "There was a urn in the room, and I was sitting on it. I was like, \'What the hell is going on?\' And he said, \'Well, I\'m not sure. I\'m just going to go back to my room and get some coffee.\' And"
    );
    assert!((output[1].score.unwrap() - (-1.3545)).abs() < 1e-4);

    Ok(())
}

#[test]
fn gpt2_bad_tokens_greedy() -> anyhow::Result<()> {
    //    Resources definition
    let config_resource = Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource = Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let model_resource = Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

    let generate_config = GenerateConfig {
        max_length: Some(36),
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        do_sample: false,
        num_beams: 1,
        device: Device::Cpu,
        ..Default::default()
    };
    let model = GPT2Generator::new(generate_config)?;

    let bad_words = vec![" honeybees", " a writer"];
    let bad_word_ids = model
        .get_tokenizer()
        .encode_list(
            bad_words.as_slice(),
            512,
            &TruncationStrategy::DoNotTruncate,
            0,
        )
        .into_iter()
        .map(|tokenized_input| tokenized_input.token_ids)
        .collect::<Vec<Vec<i64>>>();

    let input_context_1 = "Hello, my name is";

    let baseline_generate_options = GenerateOptions {
        output_scores: true,
        ..Default::default()
    };
    let test_generate_options = GenerateOptions {
        bad_word_ids: Some(&bad_word_ids),
        output_scores: true,
        ..Default::default()
    };

    let baseline_output = model.generate(Some(&[input_context_1]), Some(baseline_generate_options));
    let output = model.generate(Some(&[input_context_1]), Some(test_generate_options));

    assert_eq!(baseline_output.len(), 1);
    assert_eq!(
        baseline_output[0].text,
        "Hello, my name is John. I'm a writer, and I'm writing a book. I've been writing for a long time. I was born in New York City,"
    );
    assert!((baseline_output[0].score.unwrap() - (-1.3316)).abs() < 1e-4);

    assert_eq!(output.len(), 1);
    assert_eq!(
        output[0].text,
        "Hello, my name is John. I'm a student at the University of California, Berkeley. I've been studying computer science for a year. I have a PhD in computer science"
    );
    assert!((output[0].score.unwrap() - (-1.2307)).abs() < 1e-4);

    Ok(())
}

#[test]
fn gpt2_bad_tokens_beam_search() -> anyhow::Result<()> {
    //    Resources definition
    let config_resource = Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource = Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let model_resource = Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

    let generate_config = GenerateConfig {
        max_length: Some(36),
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        do_sample: false,
        num_beams: 3,
        device: Device::Cpu,
        ..Default::default()
    };
    let model = GPT2Generator::new(generate_config)?;

    let bad_words = vec![" honeybees", " a member"];
    let bad_word_ids = model
        .get_tokenizer()
        .encode_list(
            bad_words.as_slice(),
            512,
            &TruncationStrategy::DoNotTruncate,
            0,
        )
        .into_iter()
        .map(|tokenized_input| tokenized_input.token_ids)
        .collect::<Vec<Vec<i64>>>();

    let input_context_1 = "Hello, my name is";

    let baseline_generate_options = GenerateOptions {
        output_scores: true,
        ..Default::default()
    };
    let test_generate_options = GenerateOptions {
        bad_word_ids: Some(&bad_word_ids),
        output_scores: true,
        ..Default::default()
    };

    let baseline_output = model.generate(Some(&[input_context_1]), Some(baseline_generate_options));
    let output = model.generate(Some(&[input_context_1]), Some(test_generate_options));

    assert_eq!(baseline_output.len(), 1);
    assert_eq!(
        baseline_output[0].text,
        "Hello, my name is John, and I am a member of the Church of Jesus Christ of Latter-day Saints.\n\nI am a Mormon. I have been a member"
    );
    assert!((baseline_output[0].score.unwrap() - (-1.0503)).abs() < 1e-4);

    assert_eq!(output.len(), 1);
    assert_eq!(
        output[0].text,
        "Hello, my name is John, and I am the owner of a small business.\n\nI have been in business for over 20 years. I have been a business owner for"
    );
    assert!((output[0].score.unwrap() - (-1.3742)).abs() < 1e-4);

    Ok(())
}

#[test]
fn gpt2_prefix_allowed_token_beam_search() -> anyhow::Result<()> {
    //    Resources definition
    let config_resource = Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource = Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let model_resource = Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

    fn force_one_paragraph(_batch_id: i64, previous_token_ids: &Tensor) -> Vec<i64> {
        let paragraph_tokens = [198, 628];

        for paragraph_token in paragraph_tokens.iter() {
            if previous_token_ids
                .iter::<i64>()
                .unwrap()
                .any(|x| x == *paragraph_token)
            {
                return vec![50256];
            }
        }
        (0..50255).collect()
    }

    let generate_config = GenerateConfig {
        max_length: Some(32),
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        do_sample: false,
        num_beams: 3,
        device: Device::Cpu,
        ..Default::default()
    };
    let model = GPT2Generator::new(generate_config)?;

    let input_context_1 = "Rust is a";
    let input_context_2 = "There was a ";

    let generate_options = GenerateOptions {
        prefix_allowed_tokens_fn: Some(&force_one_paragraph),
        output_scores: true,
        ..Default::default()
    };

    let output = model.generate(
        Some(&[input_context_1, input_context_2]),
        Some(generate_options),
    );

    assert_eq!(output.len(), 2);
    assert_eq!(
        output[0].text,
        "Rust is a simple, fast, and easy-to-use framework for building web applications. It is designed to be easy to use and maintain, and"
    );
    assert!((output[0].score.unwrap() - (-1.2750)).abs() < 1e-4);
    assert_eq!(
        output[1].text,
        "There was a urn in the back of the room, and I was sitting on it, and it looked like it was going to explode. And then I"
    );
    assert!((output[1].score.unwrap() - (-1.3326)).abs() < 1e-4);

    Ok(())
}

#[test]
fn gpt2_greedy_token_scores() -> anyhow::Result<()> {
    //    Resources definition
    let config_resource = Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource = Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let model_resource = Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

    let generate_config = GenerateConfig {
        max_length: Some(16),
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        do_sample: false,
        num_beams: 1,
        device: Device::Cpu,
        ..Default::default()
    };
    let model = GPT2Generator::new(generate_config)?;

    let input_context_1 = "Hello, my name is";
    let input_context_2 = "It is a beautiful";

    let generate_options = GenerateOptions {
        output_scores: true,
        ..Default::default()
    };

    let output = model.generate_indices(
        Some(&[input_context_1, input_context_2]),
        Some(generate_options),
    );

    assert_eq!(output.len(), 2);
    assert_eq!(
        output[0].indices,
        vec![15496, 11, 616, 1438, 318, 1757, 13, 314, 1101, 257, 6260, 11, 290, 314, 1101, 3597,]
    );
    assert!((output[0].score.unwrap() - (-1.3794)).abs() < 1e-4);
    assert!((output[0].token_scores.as_ref().unwrap()[0] - (-4.6114)).abs() < 1e-4);
    assert!((output[0].token_scores.as_ref().unwrap()[1] - (-2.1742)).abs() < 1e-4);
    assert!((output[0].token_scores.as_ref().unwrap()[2] - (-0.7571)).abs() < 1e-4);

    assert_eq!(
        output[1].indices,
        vec![50256, 1026, 318, 257, 4950, 1517, 284, 766, 13, 632, 318, 257, 845, 4950, 1517, 13]
    );
    assert!((output[1].score.unwrap() - (-1.0609)).abs() < 1e-4);
    assert!((output[1].token_scores.as_ref().unwrap()[0] - (-2.6287)).abs() < 1e-4);
    assert!((output[1].token_scores.as_ref().unwrap()[1] - (-1.3033)).abs() < 1e-4);
    assert!((output[1].token_scores.as_ref().unwrap()[2] - (-0.6780)).abs() < 1e-4);

    Ok(())
}

#[test]
fn gpt2_beam_search_token_scores() -> anyhow::Result<()> {
    //    Resources definition
    let config_resource = Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource = Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let model_resource = Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

    let generate_config = GenerateConfig {
        max_length: Some(16),
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        do_sample: false,
        num_beams: 2,
        device: Device::Cpu,
        ..Default::default()
    };
    let model = GPT2Generator::new(generate_config)?;

    let input_context_1 = "Hello, my name is";
    let input_context_2 = "It is a beautiful";

    let generate_options = GenerateOptions {
        output_scores: true,
        ..Default::default()
    };

    let output = model.generate_indices(
        Some(&[input_context_1, input_context_2]),
        Some(generate_options),
    );

    assert_eq!(output.len(), 2);
    assert_eq!(
        output[0].indices,
        vec![15496, 11, 616, 1438, 318, 1757, 11, 290, 314, 716, 257, 2888, 286, 262, 1578, 1829,]
    );
    assert!((output[0].score.unwrap() - (-1.1913)).abs() < 1e-4);
    assert!((output[0].token_scores.as_ref().unwrap()[0] - (-4.6114)).abs() < 1e-4);
    assert!((output[0].token_scores.as_ref().unwrap()[1] - (-2.1742)).abs() < 1e-4);
    assert!((output[0].token_scores.as_ref().unwrap()[2] - (-0.7571)).abs() < 1e-4);

    assert_eq!(
        output[1].indices,
        vec![50256, 1026, 318, 257, 4950, 1517, 284, 766, 13, 632, 318, 257, 845, 4950, 1517, 13]
    );
    assert!((output[1].score.unwrap() - (-1.1160)).abs() < 1e-4);
    assert!((output[1].token_scores.as_ref().unwrap()[0] - (-2.6287)).abs() < 1e-4);
    assert!((output[1].token_scores.as_ref().unwrap()[1] - (-1.3033)).abs() < 1e-4);
    assert!((output[1].token_scores.as_ref().unwrap()[2] - (-0.6780)).abs() < 1e-4);

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
        max_length: Some(36),
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
    let conversation_2_id = conversation_manager.create("Hello how are you?");

    // Turn 1
    let output = conversation_model.generate_responses(&mut conversation_manager);
    assert_eq!(output.len(), 2);
    assert_eq!(output.get(&conversation_1_id).unwrap(), &"The Big Lebowski");
    assert_eq!(
        output.get(&conversation_2_id).unwrap(),
        &"I'm good, how are you?"
    );

    // Turn 2
    let _ = conversation_manager
        .get(&conversation_1_id)
        .unwrap()
        .add_user_input("Is it an action movie?");
    let _ = conversation_manager
        .get(&conversation_2_id)
        .unwrap()
        .add_user_input("Fine.");

    let output = conversation_model.generate_responses(&mut conversation_manager);
    assert_eq!(output.len(), 2);
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
