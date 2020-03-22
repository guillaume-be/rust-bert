extern crate failure;
extern crate dirs;

use std::path::PathBuf;
use tch::{Device, nn, Tensor, no_grad};
use rust_tokenizers::{BertTokenizer, TruncationStrategy, Tokenizer, Vocab};
use rust_bert::Config;
use rust_bert::bert::{BertConfig, BertForMaskedLM, BertForSequenceClassification, BertForMultipleChoice, BertForTokenClassification, BertForQuestionAnswering};
use rust_bert::pipelines::ner::NERModel;


#[test]
fn bert_masked_lm() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("bert");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let weights_path = &home.as_path().join("model.ot");

//    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
    let config = BertConfig::from_file(config_path);
    let bert_model = BertForMaskedLM::new(&vs.root(), &config);
    vs.load(weights_path)?;

//    Define input
    let input = ["Looks like one thing is missing", "It\'s like comparing oranges to apples"];
    let tokenized_input = tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
    let max_len = tokenized_input.iter().map(|input| input.token_ids.len()).max().unwrap();
    let mut tokenized_input = tokenized_input.
        iter().
        map(|input| input.token_ids.clone()).
        map(|mut input| {
            input.extend(vec![0; max_len - input.len()]);
            input
        }).
        collect::<Vec<_>>();

//    Masking the token [thing] of sentence 1 and [oranges] of sentence 2
    tokenized_input[0][4] = 103;
    tokenized_input[1][6] = 103;
    let tokenized_input = tokenized_input.
        iter().
        map(|input|
            Tensor::of_slice(&(input))).
        collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

//    Forward pass
    let (output, _, _) = no_grad(|| {
        bert_model
            .forward_t(Some(input_tensor),
                       None,
                       None,
                       None,
                       None,
                       &None,
                       &None,
                       false)
    });

//    Print masked tokens
    let index_1 = output.get(0).get(4).argmax(0, false);
    let index_2 = output.get(1).get(6).argmax(0, false);
    let word_1 = tokenizer.vocab().id_to_token(&index_1.int64_value(&[]));
    let word_2 = tokenizer.vocab().id_to_token(&index_2.int64_value(&[]));

    assert_eq!("person", word_1); // Outputs "person" : "Looks like one [person] is missing"
    assert_eq!("orange", word_2);// Outputs "pear" : "It\'s like comparing [pear] to apples"

    Ok(())
}

#[test]
fn bert_for_sequence_classification() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("bert");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");


//    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
    let mut config = BertConfig::from_file(config_path);
    config.num_labels = Some(42);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let bert_model = BertForSequenceClassification::new(&vs.root(), &config);


//    Define input
    let input = ["Looks like one thing is missing", "It\'s like comparing oranges to apples"];
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
    let (output, all_hidden_states, all_attentions) = no_grad(|| {
        bert_model
            .forward_t(Some(input_tensor),
                       None,
                       None,
                       None,
                       None,
                       false)
    });

    assert_eq!(output.size(), &[2, 42]);
    assert_eq!(config.num_hidden_layers as usize, all_hidden_states.unwrap().len());
    assert_eq!(config.num_hidden_layers as usize, all_attentions.unwrap().len());

    Ok(())
}

#[test]
fn bert_for_multiple_choice() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("bert");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");


//    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
    let mut config = BertConfig::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let bert_model = BertForMultipleChoice::new(&vs.root(), &config);


//    Define input
    let input = ["Looks like one thing is missing", "It\'s like comparing oranges to apples"];
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
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device).unsqueeze(0);

//    Forward pass
    let (output, all_hidden_states, all_attentions) = no_grad(|| {
        bert_model
            .forward_t(input_tensor,
                       None,
                       None,
                       None,
                       None,
                       false)
    });

    assert_eq!(output.size(), &[1, 2]);
    assert_eq!(config.num_hidden_layers as usize, all_hidden_states.unwrap().len());
    assert_eq!(config.num_hidden_layers as usize, all_attentions.unwrap().len());

    Ok(())
}

#[test]
fn bert_for_token_classification() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("bert");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");


//    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
    let mut config = BertConfig::from_file(config_path);
    config.num_labels = Some(7);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let bert_model = BertForTokenClassification::new(&vs.root(), &config);


//    Define input
    let input = ["Looks like one thing is missing", "It\'s like comparing oranges to apples"];
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
    let (output, all_hidden_states, all_attentions) = no_grad(|| {
        bert_model
            .forward_t(Some(input_tensor),
                       None,
                       None,
                       None,
                       None,
                       false)
    });

    assert_eq!(output.size(), &[2, 11, 7]);
    assert_eq!(config.num_hidden_layers as usize, all_hidden_states.unwrap().len());
    assert_eq!(config.num_hidden_layers as usize, all_attentions.unwrap().len());

    Ok(())
}

#[test]
fn bert_for_question_answering() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("bert");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");


//    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true);
    let mut config = BertConfig::from_file(config_path);
    config.num_labels = Some(7);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let bert_model = BertForQuestionAnswering::new(&vs.root(), &config);


//    Define input
    let input = ["Looks like one thing is missing", "It\'s like comparing oranges to apples"];
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
    let (start_scores, end_scores, all_hidden_states, all_attentions) = no_grad(|| {
        bert_model
            .forward_t(Some(input_tensor),
                       None,
                       None,
                       None,
                       None,
                       false)
    });

    assert_eq!(start_scores.size(), &[2, 11]);
    assert_eq!(end_scores.size(), &[2, 11]);
    assert_eq!(config.num_hidden_layers as usize, all_hidden_states.unwrap().len());
    assert_eq!(config.num_hidden_layers as usize, all_attentions.unwrap().len());

    Ok(())
}

#[test]
fn bert_pre_trained_ner() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("bert-ner");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let weights_path = &home.as_path().join("model.ot");

//    Set-up model
    let device = Device::cuda_if_available();
    let ner_model = NERModel::new(vocab_path,
                                  config_path,
                                  weights_path, device)?;

//    Define input
    let input = [
        "My name is Amy. I live in Paris.",
        "Paris is a city in France."
    ];

//    Run model
    let output = ner_model.predict(&input);


    assert_eq!(output.len(), 4);

    assert_eq!(output[0].word, "Amy");
    assert!((output[0].score - 0.9986).abs() < 1e-4);
    assert_eq!(output[0].label, "I-PER");

    assert_eq!(output[1].word, "Paris");
    assert!((output[1].score - 0.9986).abs() < 1e-4);
    assert_eq!(output[1].label, "I-LOC");

    assert_eq!(output[2].word, "Paris");
    assert!((output[2].score - 0.9988).abs() < 1e-4);
    assert_eq!(output[2].label, "I-LOC");

    assert_eq!(output[3].word, "France");
    assert!((output[3].score - 0.9994).abs() < 1e-4);
    assert_eq!(output[3].label, "I-LOC");

    Ok(())
}