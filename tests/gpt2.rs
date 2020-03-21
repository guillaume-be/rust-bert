use std::path::PathBuf;
use tch::{Device, nn, Tensor};
use rust_tokenizers::{Gpt2Tokenizer, TruncationStrategy, Tokenizer};
use rust_bert::{Gpt2Config, GPT2LMHeadModel, Config, LMHeadModel, GPT2Generator, LanguageGenerator};

#[test]
fn gpt2_lm_model() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("gpt2");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let merges_path = &home.as_path().join("merges.txt");
    let weights_path = &home.as_path().join("model.ot");

//    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_file(vocab_path.to_str().unwrap(), merges_path.to_str().unwrap(), false);
    let config = Gpt2Config::from_file(config_path);
    let gpt2_model = GPT2LMHeadModel::new(&vs.root(), &config);
    vs.load(weights_path)?;

//    Define input
    let input = ["One two three four"];
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
    let (output, past, _, _) = gpt2_model.forward_t(
        &Some(input_tensor),
        &None,
        &None,
        &None,
        &None,
        &None,
        false).unwrap();

    let next_word_id = output.get(0).get(-1).argmax(-1, true).int64_value(&[0]);
    let next_word = tokenizer.decode(vec!(next_word_id), true, true);

    assert_eq!(output.size(), vec!(1, 4, 50257));
    assert!(past.is_some());
    assert_eq!(past.as_ref().unwrap().len(), config.n_layer as usize);
    assert_eq!(past.as_ref().unwrap()[0].size(), vec!(2, 1, config.n_head, 4, 64));
    assert!((output.double_value(&[0, output.size()[1] - 1, next_word_id]) - (-69.4948)).abs() < 1e-4);
    assert_eq!(next_word_id, 1936i64);
    assert_eq!(next_word, String::from(" five"));

    Ok(())
}


#[test]
fn gpt2_generation_greedy() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("gpt2");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let merges_path = &home.as_path().join("merges.txt");
    let weights_path = &home.as_path().join("model.ot");

//    Set-up masked LM model
    let device = Device::cuda_if_available();

//    let model = OpenAIGenerator::new(vocab_path, merges_path, config_path, weights_path, device)?;
    let model = GPT2Generator::new(vocab_path, merges_path, config_path, weights_path, device)?;

    let input_context = "The cat";
    let output = model.generate(Some(vec!(input_context)), 0, 40, false, false, 1, 1.0,
                                 0, 0.9, 1.1, 1.0, 3, 1, None);

    assert_eq!(output.len(), 1);
    assert_eq!(output[0], "The cat was found in a field near the town of Keflavik, about 30 miles (48 kilometers) south-east of Moscow.\n\n\n");

    Ok(())
}

#[test]
fn gpt2_generation_beam_search() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("gpt2");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let merges_path = &home.as_path().join("merges.txt");
    let weights_path = &home.as_path().join("model.ot");

//    Set-up masked LM model
    let device = Device::cuda_if_available();

//    let model = OpenAIGenerator::new(vocab_path, merges_path, config_path, weights_path, device)?;
    let model = GPT2Generator::new(vocab_path, merges_path, config_path, weights_path, device)?;

    let input_context = "The dog";
    let output = model.generate(Some(vec!(input_context)), 0, 20, false, false, 5, 1.2,
                                 0, 0.9, 1.0, 1.0, 3, 3, None);

    assert_eq!(output.len(), 3);
    assert_eq!(output[0], "The dog was found in the backyard of a home in the 6200 block of South Main Street.");
    assert_eq!(output[1], "The dog was found in the backyard of a home in the 6500 block of South Main Street.");
    assert_eq!(output[2], "The dog was found in the backyard of a home in the 6200 block of South Main Street,");

    Ok(())
}

#[test]
fn gpt2_generation_beam_search_multiple_prompts_without_padding() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("gpt2");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let merges_path = &home.as_path().join("merges.txt");
    let weights_path = &home.as_path().join("model.ot");

//    Set-up masked LM model
    let device = Device::cuda_if_available();

//    let model = OpenAIGenerator::new(vocab_path, merges_path, config_path, weights_path, device)?;
    let model = GPT2Generator::new(vocab_path, merges_path, config_path, weights_path, device)?;

    let input_context_1 = "The dog";
    let input_context_2 = "The cat";
    let output = model.generate(Some(vec!(input_context_1, input_context_2)), 0, 20, false, false, 5, 1.2,
                                 0, 0.9, 1.0, 1.0, 3, 3, None);

    assert_eq!(output.len(), 6);
    assert_eq!(output[0], "The dog was found in the backyard of a home in the 6200 block of South Main Street.");
    assert_eq!(output[1], "The dog was found in the backyard of a home in the 6500 block of South Main Street.");
    assert_eq!(output[2], "The dog was found in the backyard of a home in the 6200 block of South Main Street,");
    assert_eq!(output[3], "The cat-and-mouse game.\n\n\"I think it\'s going to be interesting to");
    assert_eq!(output[4], "The cat-and-mouse game.\n\n\"I think it\'s going to be a very");
    assert_eq!(output[5], "The cat-and-mouse game.\n\n\"I think it\'s going to be very interesting");

    Ok(())
}

#[test]
fn gpt2_generation_beam_search_multiple_prompts_with_padding() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("gpt2");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let merges_path = &home.as_path().join("merges.txt");
    let weights_path = &home.as_path().join("model.ot");

//    Set-up masked LM model
    let device = Device::cuda_if_available();

//    let model = OpenAIGenerator::new(vocab_path, merges_path, config_path, weights_path, device)?;
    let model = GPT2Generator::new(vocab_path, merges_path, config_path, weights_path, device)?;

    let input_context_1 = "The dog";
    let input_context_2 = "The cat was";
    let output = model.generate(Some(vec!(input_context_1, input_context_2)), 0, 20, false, false, 5, 1.2,
                                 0, 0.9, 1.0, 1.0, 3, 3, None);

    assert_eq!(output.len(), 6);
    assert_eq!(output[0], "The dog was found dead on the side of the road in the middle of the night.\n");
    assert_eq!(output[1], "The dog was found dead on the side of the road in the middle of the night on Sunday");
    assert_eq!(output[2], "The dog was found dead on the side of the road in the middle of the night on Saturday");
    assert_eq!(output[3], "The cat was taken to a local hospital, where it was treated and released.\n\nPolice said");
    assert_eq!(output[4], "The cat was taken to a local hospital, where it was treated and released.\n\n\"It");
    assert_eq!(output[5], "The cat was taken to a local hospital, where it was treated and released.\n\n\"We");

    Ok(())
}