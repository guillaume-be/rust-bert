use rust_bert::deberta_v2::{
    DebertaV2Config, DebertaV2ConfigResources, DebertaV2ForMaskedLM, DebertaV2ForQuestionAnswering,
    DebertaV2ForSequenceClassification, DebertaV2ForTokenClassification, DebertaV2VocabResources,
};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{DeBERTaV2Tokenizer, MultiThreadedTokenizer, TruncationStrategy};
use std::collections::HashMap;
use tch::{nn, no_grad, Device, Kind, Tensor};

extern crate anyhow;

#[test]
fn deberta_v2_masked_lm() -> anyhow::Result<()> {
    //    Set-up masked LM model
    let config_resource = Box::new(RemoteResource::from_pretrained(
        DebertaV2ConfigResources::DEBERTA_V3_BASE,
    ));
    let config_path = config_resource.get_local_path()?;
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let mut config = DebertaV2Config::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let deberta_model = DebertaV2ForMaskedLM::new(vs.root(), &config);

    //    Generate random input
    let input_tensor = Tensor::randint(42, [32, 128], (Kind::Int64, device));
    let attention_mask = Tensor::ones([32, 128], (Kind::Int64, device));
    let position_ids = Tensor::arange(128, (Kind::Int64, device)).unsqueeze(0);
    let token_type_ids = Tensor::zeros([32, 128], (Kind::Int64, device));

    //    Forward pass
    let model_output = no_grad(|| {
        deberta_model.forward_t(
            Some(&input_tensor),
            Some(&attention_mask),
            Some(&token_type_ids),
            Some(&position_ids),
            None,
            false,
        )
    })?;

    assert_eq!(model_output.logits.size(), vec!(32, 128, config.vocab_size));
    assert!(model_output.all_attentions.is_some());
    assert!(model_output.all_hidden_states.is_some());
    assert_eq!(
        config.num_hidden_layers as usize,
        model_output.all_hidden_states.as_ref().unwrap().len()
    );
    assert_eq!(
        config.num_hidden_layers as usize,
        model_output.all_attentions.as_ref().unwrap().len()
    );
    assert_eq!(
        model_output.all_attentions.as_ref().unwrap()[0].size(),
        vec!(32, 12, 128, 128)
    );
    assert_eq!(
        model_output.all_hidden_states.as_ref().unwrap()[0].size(),
        vec!(32, 128, config.hidden_size)
    );

    Ok(())
}

#[test]
fn deberta_v2_for_sequence_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        DebertaV2ConfigResources::DEBERTA_V3_BASE,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        DebertaV2VocabResources::DEBERTA_V3_BASE,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let tokenizer =
        DeBERTaV2Tokenizer::from_file(vocab_path.to_str().unwrap(), false, false, false)?;
    let mut config = DebertaV2Config::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("Positive"));
    dummy_label_mapping.insert(1, String::from("Neutral"));
    dummy_label_mapping.insert(2, String::from("Negative"));
    config.id2label = Some(dummy_label_mapping);
    let model = DebertaV2ForSequenceClassification::new(vs.root(), &config)?;

    //    Define input
    let inputs = ["Where's Paris?", "In Kentucky, United States"];
    let tokenized_input = tokenizer.encode_list(&inputs, 128, &TruncationStrategy::LongestFirst, 0);
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
        no_grad(|| model.forward_t(Some(input_tensor.as_ref()), None, None, None, None, false))?;

    assert_eq!(model_output.logits.size(), &[2, 3]);

    Ok(())
}

#[test]
fn deberta_v2_for_token_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        DebertaV2ConfigResources::DEBERTA_V3_BASE,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        DebertaV2VocabResources::DEBERTA_V3_BASE,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let tokenizer =
        DeBERTaV2Tokenizer::from_file(vocab_path.to_str().unwrap(), false, false, false)?;
    let mut config = DebertaV2Config::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("O"));
    dummy_label_mapping.insert(1, String::from("LOC"));
    dummy_label_mapping.insert(2, String::from("PER"));
    dummy_label_mapping.insert(3, String::from("ORG"));
    config.id2label = Some(dummy_label_mapping);
    let model = DebertaV2ForTokenClassification::new(vs.root(), &config)?;

    //    Define input
    let inputs = ["Where's Paris?", "In Kentucky, United States"];
    let tokenized_input = tokenizer.encode_list(&inputs, 128, &TruncationStrategy::LongestFirst, 0);
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
        no_grad(|| model.forward_t(Some(input_tensor.as_ref()), None, None, None, None, false))?;

    assert_eq!(model_output.logits.size(), &[2, 7, 4]);

    Ok(())
}

#[test]
fn deberta_v2_for_question_answering() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        DebertaV2ConfigResources::DEBERTA_V3_BASE,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        DebertaV2VocabResources::DEBERTA_V3_BASE,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let tokenizer =
        DeBERTaV2Tokenizer::from_file(vocab_path.to_str().unwrap(), false, false, false)?;
    let config = DebertaV2Config::from_file(config_path);
    let model = DebertaV2ForQuestionAnswering::new(vs.root(), &config);

    //    Define input
    let inputs = ["Where's Paris?", "Paris is in In Kentucky, United States"];
    let tokenized_input = tokenizer.encode_pair_list(
        &[(inputs[0], inputs[1])],
        128,
        &TruncationStrategy::LongestFirst,
        0,
    );
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
        no_grad(|| model.forward_t(Some(input_tensor.as_ref()), None, None, None, None, false))?;

    assert_eq!(model_output.start_logits.size(), &[1, 16]);
    assert_eq!(model_output.end_logits.size(), &[1, 16]);
    Ok(())
}
