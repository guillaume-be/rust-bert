use std::path::PathBuf;
use tch::{Device, nn, Tensor};
use rust_tokenizers::{TruncationStrategy, Tokenizer, OpenAiGptTokenizer};
use rust_bert::gpt2::gpt2::{Gpt2Config, LMHeadModel};
use rust_bert::common::config::Config;
use rust_bert::openai_gpt::openai_gpt::OpenAIGPTLMHeadModel;
use rust_bert::pipelines::generation::{OpenAIGenerator, LanguageGenerator};

#[test]
fn openai_gpt_lm_model() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("openai-gpt");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let merges_path = &home.as_path().join("merges.txt");
    let weights_path = &home.as_path().join("model.ot");

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
    let (output, _, _, _) = openai_gpt.forward_t(
        &Some(input_tensor),
        &None,
        &None,
        &None,
        &None,
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
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("openai-gpt");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let merges_path = &home.as_path().join("merges.txt");
    let weights_path = &home.as_path().join("model.ot");

//    Set-up masked LM model
    let device = Device::cuda_if_available();

//    let model = OpenAIGenerator::new(vocab_path, merges_path, config_path, weights_path, device)?;
    let model = OpenAIGenerator::new(vocab_path, merges_path, config_path, weights_path, device)?;

    let input_context = "It was an intense machine dialogue. ";
    let output = model.generate(Some(input_context), 0, 40, false, false, 1, 1.0,
                                 0, 1.0, 1.0, 1.0, 0, 1, None);

    assert_eq!(output.len(), 1);
    assert_eq!(output[0], "it was an intense machine dialogue. \n \" i 'm sorry, \" i said. \" i 'm not sure what you're talking about. \" \n \" you're not a vampire, \" he said");

    Ok(())
}

#[test]
fn openai_gpt_generation_beam_search() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("openai-gpt");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let merges_path = &home.as_path().join("merges.txt");
    let weights_path = &home.as_path().join("model.ot");

//    Set-up masked LM model
    let device = Device::cuda_if_available();

//    let model = OpenAIGenerator::new(vocab_path, merges_path, config_path, weights_path, device)?;
    let model = OpenAIGenerator::new(vocab_path, merges_path, config_path, weights_path, device)?;

    let input_context = "What?!";
    let output = model.generate(Some(input_context), 0, 20, false, false, 5, 2.0,
                                 0, 1.0, 1.0, 1.0, 0, 3, None);

    assert_eq!(output.len(), 3);
    assert_eq!(output[0], "what?! \" i yelled. \" what are you talking about? i don't know what you");
    assert_eq!(output[1], "what?! \" i yelled. \" what are you talking about? i don't even know what");
    assert_eq!(output[2], "what?! \" i yelled. \" what are you talking about? i don't understand what you");

    Ok(())
}