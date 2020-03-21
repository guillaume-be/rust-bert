use std::path::PathBuf;
use tch::{Device, nn, Tensor};
use rust_tokenizers::{Gpt2Tokenizer, TruncationStrategy, Tokenizer};
use rust_bert::{Gpt2Config, GPT2LMHeadModel, Config, LMHeadModel};

#[test]
fn distilgpt2_lm_model() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("distilgpt2");
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
    let input = ["One two three four five six seven eight nine ten eleven"];
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

    assert_eq!(output.size(), vec!(1, 11, 50257));
    assert!(past.is_some());
    assert_eq!(past.as_ref().unwrap().len(), config.n_layer as usize);
    assert_eq!(past.as_ref().unwrap()[0].size(), vec!(2, 1, config.n_head, 11, 64));
    assert!((output.double_value(&[0, output.size()[1] - 1, next_word_id]) - (-48.7065)).abs() < 1e-4);
    assert_eq!(next_word_id, 14104i64);
    assert_eq!(next_word, String::from(" twelve"));

    Ok(())
}