extern crate anyhow;
extern crate dirs;

use rust_bert::albert::{
    AlbertConfig, AlbertConfigResources, AlbertForMaskedLM, AlbertForMultipleChoice,
    AlbertForQuestionAnswering, AlbertForSequenceClassification, AlbertForTokenClassification,
    AlbertModelResources, AlbertVocabResources,
};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{AlbertTokenizer, MultiThreadedTokenizer, TruncationStrategy};
use rust_tokenizers::vocab::Vocab;
use std::collections::HashMap;
use tch::{nn, no_grad, Device, Tensor};

#[test]
fn albert_masked_lm() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        AlbertConfigResources::ALBERT_BASE_V2,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        AlbertVocabResources::ALBERT_BASE_V2,
    ));
    let weights_resource = Box::new(RemoteResource::from_pretrained(
        AlbertModelResources::ALBERT_BASE_V2,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: AlbertTokenizer =
        AlbertTokenizer::from_file(vocab_path.to_str().unwrap(), true, false)?;
    let config = AlbertConfig::from_file(config_path);
    let albert_model = AlbertForMaskedLM::new(vs.root(), &config);
    vs.load(weights_path)?;

    //    Define input
    let input = [
        "Looks like one [MASK] is missing",
        "It\'s like comparing [MASK] to apples",
    ];
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
        no_grad(|| albert_model.forward_t(Some(&input_tensor), None, None, None, None, false));

    //    Print masked tokens
    let index_1 = model_output
        .prediction_scores
        .get(0)
        .get(4)
        .argmax(0, false);
    let index_2 = model_output
        .prediction_scores
        .get(1)
        .get(6)
        .argmax(0, false);
    let word_1 = tokenizer.vocab().id_to_token(&index_1.int64_value(&[]));
    let word_2 = tokenizer.vocab().id_to_token(&index_2.int64_value(&[]));

    assert_eq!("▁them", word_1); // Outputs "_them" : "Looks like one [them] is missing (? this is identical with the original implementation)"
    assert_eq!("▁grapes", word_2); // Outputs "grapes" : "It\'s like comparing [grapes] to apples"
    assert!((model_output.prediction_scores.double_value(&[0, 0, 0]) - 4.6143).abs() < 1e-4);
    Ok(())
}

#[test]
fn albert_for_sequence_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        AlbertConfigResources::ALBERT_BASE_V2,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        AlbertVocabResources::ALBERT_BASE_V2,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: AlbertTokenizer =
        AlbertTokenizer::from_file(vocab_path.to_str().unwrap(), true, false)?;
    let mut config = AlbertConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("Positive"));
    dummy_label_mapping.insert(1, String::from("Negative"));
    dummy_label_mapping.insert(3, String::from("Neutral"));
    config.id2label = Some(dummy_label_mapping);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let albert_model = AlbertForSequenceClassification::new(vs.root(), &config)?;

    //    Define input
    let input = [
        "Looks like one thing is missing",
        "It\'s like comparing oranges to apples",
    ];
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
        no_grad(|| albert_model.forward_t(Some(&input_tensor), None, None, None, None, false));

    assert_eq!(model_output.logits.size(), &[2, 3]);
    assert_eq!(
        config.num_hidden_layers as usize,
        model_output.all_hidden_states.unwrap().len()
    );
    assert_eq!(
        config.num_hidden_layers as usize,
        model_output.all_attentions.unwrap().len()
    );

    Ok(())
}

#[test]
fn albert_for_multiple_choice() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        AlbertConfigResources::ALBERT_BASE_V2,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        AlbertVocabResources::ALBERT_BASE_V2,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: AlbertTokenizer =
        AlbertTokenizer::from_file(vocab_path.to_str().unwrap(), true, false)?;
    let mut config = AlbertConfig::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let albert_model = AlbertForMultipleChoice::new(vs.root(), &config);

    //    Define input
    let input = [
        "Looks like one thing is missing",
        "It\'s like comparing oranges to apples",
    ];
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
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0)
        .to(device)
        .unsqueeze(0);

    //    Forward pass
    let model_output = no_grad(|| {
        albert_model
            .forward_t(Some(&input_tensor), None, None, None, None, false)
            .unwrap()
    });

    assert_eq!(model_output.logits.size(), &[1, 2]);
    assert_eq!(
        config.num_hidden_layers as usize,
        model_output.all_hidden_states.unwrap().len()
    );
    assert_eq!(
        config.num_hidden_layers as usize,
        model_output.all_attentions.unwrap().len()
    );

    Ok(())
}

#[test]
fn albert_for_token_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        AlbertConfigResources::ALBERT_BASE_V2,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        AlbertVocabResources::ALBERT_BASE_V2,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: AlbertTokenizer =
        AlbertTokenizer::from_file(vocab_path.to_str().unwrap(), true, false)?;
    let mut config = AlbertConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("O"));
    dummy_label_mapping.insert(1, String::from("LOC"));
    dummy_label_mapping.insert(2, String::from("PER"));
    dummy_label_mapping.insert(3, String::from("ORG"));
    config.id2label = Some(dummy_label_mapping);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let albert_model = AlbertForTokenClassification::new(vs.root(), &config)?;

    //    Define input
    let input = [
        "Looks like one thing is missing",
        "It\'s like comparing oranges to apples",
    ];
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
        no_grad(|| albert_model.forward_t(Some(&input_tensor), None, None, None, None, false));

    assert_eq!(model_output.logits.size(), &[2, 12, 4]);
    assert_eq!(
        config.num_hidden_layers as usize,
        model_output.all_hidden_states.unwrap().len()
    );
    assert_eq!(
        config.num_hidden_layers as usize,
        model_output.all_attentions.unwrap().len()
    );

    Ok(())
}

#[test]
fn albert_for_question_answering() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        AlbertConfigResources::ALBERT_BASE_V2,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        AlbertVocabResources::ALBERT_BASE_V2,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: AlbertTokenizer =
        AlbertTokenizer::from_file(vocab_path.to_str().unwrap(), true, false)?;
    let mut config = AlbertConfig::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let albert_model = AlbertForQuestionAnswering::new(vs.root(), &config);

    //    Define input
    let input = [
        "Looks like one thing is missing",
        "It\'s like comparing oranges to apples",
    ];
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
        no_grad(|| albert_model.forward_t(Some(&input_tensor), None, None, None, None, false));

    assert_eq!(model_output.start_logits.size(), &[2, 12]);
    assert_eq!(model_output.end_logits.size(), &[2, 12]);
    assert_eq!(
        config.num_hidden_layers as usize,
        model_output.all_hidden_states.unwrap().len()
    );
    assert_eq!(
        config.num_hidden_layers as usize,
        model_output.all_attentions.unwrap().len()
    );

    Ok(())
}
