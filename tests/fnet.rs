extern crate anyhow;
extern crate dirs;

use rust_bert::fnet::{
    FNetConfig, FNetConfigResources, FNetForMaskedLM, FNetForMultipleChoice,
    FNetForQuestionAnswering, FNetForTokenClassification, FNetModelResources, FNetVocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::sentiment::{SentimentConfig, SentimentModel, SentimentPolarity};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{FNetTokenizer, MultiThreadedTokenizer, TruncationStrategy};
use rust_tokenizers::vocab::Vocab;
use std::collections::HashMap;
use std::convert::TryFrom;
use tch::{nn, no_grad, Device, Tensor};

#[test]
fn fnet_masked_lm() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(FNetConfigResources::BASE));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(FNetVocabResources::BASE));
    let weights_resource = Box::new(RemoteResource::from_pretrained(FNetModelResources::BASE));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: FNetTokenizer =
        FNetTokenizer::from_file(vocab_path.to_str().unwrap(), false, false)?;
    let config = FNetConfig::from_file(config_path);
    let fnet_model = FNetForMaskedLM::new(vs.root(), &config);
    vs.load(weights_path)?;

    //    Define input
    let input = [
        "Looks like one [MASK] is missing",
        "It was a very nice and [MASK] day",
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
            input.extend(vec![3; max_len - input.len()]);
            input
        })
        .map(|input| Tensor::from_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let model_output =
        no_grad(|| fnet_model.forward_t(Some(&input_tensor), None, None, None, false))?;

    //    Print masked tokens
    let index_1 = model_output
        .prediction_scores
        .get(0)
        .get(4)
        .argmax(0, false);
    let index_2 = model_output
        .prediction_scores
        .get(1)
        .get(7)
        .argmax(0, false);
    let word_1 = tokenizer.vocab().id_to_token(&index_1.int64_value(&[]));
    let word_2 = tokenizer.vocab().id_to_token(&index_2.int64_value(&[]));

    assert_eq!("▁one", word_1);
    assert_eq!("▁the", word_2);
    let value = (f64::try_from(model_output.prediction_scores.get(0).get(4).max()).unwrap()
        - 13.1721)
        .abs();
    dbg!(value);
    assert!(value < 1e-3);
    Ok(())
}

#[test]
fn fnet_for_sequence_classification() -> anyhow::Result<()> {
    // Set up classifier
    let config_resource = Box::new(RemoteResource::from_pretrained(
        FNetConfigResources::BASE_SST2,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        FNetVocabResources::BASE_SST2,
    ));
    let model_resource = Box::new(RemoteResource::from_pretrained(
        FNetModelResources::BASE_SST2,
    ));

    let sentiment_config = SentimentConfig {
        model_type: ModelType::FNet,
        model_resource,
        config_resource,
        vocab_resource,
        ..Default::default()
    };

    let sentiment_classifier = SentimentModel::new(sentiment_config)?;

    //    Get sentiments
    let input = [
        "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
        "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
        "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
    ];

    let output = sentiment_classifier.predict(input);

    assert_eq!(output.len(), 3usize);
    assert_eq!(output[0].polarity, SentimentPolarity::Negative);
    assert!((output[0].score - 0.9978).abs() < 1e-4);
    assert_eq!(output[1].polarity, SentimentPolarity::Negative);
    assert!((output[1].score - 0.9982).abs() < 1e-4);
    assert_eq!(output[2].polarity, SentimentPolarity::Positive);
    assert!((output[2].score - 0.7570).abs() < 1e-4);

    Ok(())
}

//
#[test]
fn fnet_for_multiple_choice() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(FNetConfigResources::BASE));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(FNetVocabResources::BASE));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: FNetTokenizer =
        FNetTokenizer::from_file(vocab_path.to_str().unwrap(), false, false)?;
    let mut config = FNetConfig::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let fnet_model = FNetForMultipleChoice::new(vs.root(), &config);

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
        fnet_model
            .forward_t(Some(&input_tensor), None, None, None, false)
            .unwrap()
    });

    assert_eq!(model_output.logits.size(), &[1, 2]);
    assert_eq!(
        config.num_hidden_layers as usize,
        model_output.all_hidden_states.unwrap().len()
    );

    Ok(())
}

#[test]
fn fnet_for_token_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(FNetConfigResources::BASE));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(FNetVocabResources::BASE));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: FNetTokenizer =
        FNetTokenizer::from_file(vocab_path.to_str().unwrap(), false, false)?;
    let mut config = FNetConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("O"));
    dummy_label_mapping.insert(1, String::from("LOC"));
    dummy_label_mapping.insert(2, String::from("PER"));
    dummy_label_mapping.insert(3, String::from("ORG"));
    config.id2label = Some(dummy_label_mapping);
    config.output_hidden_states = Some(true);
    let fnet_model = FNetForTokenClassification::new(vs.root(), &config)?;

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
    let model_output = no_grad(|| {
        fnet_model
            .forward_t(Some(&input_tensor), None, None, None, false)
            .unwrap()
    });

    assert_eq!(model_output.logits.size(), &[2, 11, 4]);
    assert_eq!(
        config.num_hidden_layers as usize,
        model_output.all_hidden_states.unwrap().len()
    );

    Ok(())
}

#[test]
fn fnet_for_question_answering() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(FNetConfigResources::BASE));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(FNetVocabResources::BASE));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: FNetTokenizer =
        FNetTokenizer::from_file(vocab_path.to_str().unwrap(), false, false)?;
    let mut config = FNetConfig::from_file(config_path);
    config.output_hidden_states = Some(true);
    let fnet_model = FNetForQuestionAnswering::new(vs.root(), &config);

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
    let model_output = no_grad(|| {
        fnet_model
            .forward_t(Some(&input_tensor), None, None, None, false)
            .unwrap()
    });

    assert_eq!(model_output.start_logits.size(), &[2, 11]);
    assert_eq!(model_output.end_logits.size(), &[2, 11]);
    assert_eq!(
        config.num_hidden_layers as usize,
        model_output.all_hidden_states.unwrap().len()
    );

    Ok(())
}
