use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::xlnet::{
    XLNetConfig, XLNetConfigResources, XLNetForMultipleChoice, XLNetForQuestionAnswering,
    XLNetForSequenceClassification, XLNetForTokenClassification, XLNetLMHeadModel, XLNetModel,
    XLNetModelResources, XLNetVocabResources,
};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{MultiThreadedTokenizer, TruncationStrategy, XLNetTokenizer};
use rust_tokenizers::vocab::Vocab;
use std::collections::HashMap;
use tch::{nn, no_grad, Device, Kind, Tensor};

#[test]
fn xlnet_base_model() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        XLNetConfigResources::XLNET_BASE_CASED,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        XLNetVocabResources::XLNET_BASE_CASED,
    ));
    let weights_resource = Box::new(RemoteResource::from_pretrained(
        XLNetModelResources::XLNET_BASE_CASED,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    let tokenizer: XLNetTokenizer =
        XLNetTokenizer::from_file(vocab_path.to_str().unwrap(), false, true)?;
    let mut config = XLNetConfig::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let xlnet_model = XLNetModel::new(&vs.root() / "transformer", &config);
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
        .map(|input| Tensor::from_slice(&(input[..input.len() - 2])))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    // Forward pass
    let perm_mask = Tensor::zeros([1, 4, 4], (Kind::Float, device));
    let _ = perm_mask.narrow(2, 3, 1).fill_(1.0);

    let target_mapping = Tensor::zeros([1, 1, 4], (Kind::Float, device));
    let _ = target_mapping.narrow(2, 3, 1).fill_(1.0);
    let model_output = no_grad(|| {
        xlnet_model
            .forward_t(
                Some(&input_tensor),
                None,
                None,
                Some(perm_mask.as_ref()),
                Some(target_mapping.as_ref()),
                None,
                None,
                false,
            )
            .unwrap()
    });

    assert_eq!(model_output.hidden_state.size(), vec!(1, 1, 768));
    assert!(model_output.next_cache.is_some());
    assert!(model_output.all_attentions.is_some());
    assert!(model_output.all_hidden_states.is_some());
    assert_eq!(
        config.n_layer as usize,
        model_output.all_hidden_states.as_ref().unwrap().len()
    );
    assert_eq!(
        config.n_layer as usize,
        model_output.all_attentions.as_ref().unwrap().len()
    );
    assert!(model_output.all_attentions.as_ref().unwrap()[0].1.is_some());
    assert_eq!(
        model_output.all_attentions.as_ref().unwrap()[0].0.size(),
        vec!(4, 4, 1, 12)
    );
    assert_eq!(
        model_output.all_attentions.as_ref().unwrap()[0]
            .1
            .as_ref()
            .unwrap()
            .size(),
        vec!(4, 4, 1, 12)
    );
    assert_eq!(
        model_output.all_hidden_states.as_ref().unwrap()[0].0.size(),
        vec!(4, 1, 768)
    );
    assert_eq!(
        model_output.all_hidden_states.as_ref().unwrap()[0]
            .1
            .as_ref()
            .unwrap()
            .size(),
        vec!(1, 1, 768)
    );
    Ok(())
}

#[test]
fn xlnet_lm_model() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        XLNetConfigResources::XLNET_BASE_CASED,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        XLNetVocabResources::XLNET_BASE_CASED,
    ));
    let weights_resource = Box::new(RemoteResource::from_pretrained(
        XLNetModelResources::XLNET_BASE_CASED,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    let tokenizer: XLNetTokenizer =
        XLNetTokenizer::from_file(vocab_path.to_str().unwrap(), false, true)?;
    let config = XLNetConfig::from_file(config_path);
    let xlnet_model = XLNetLMHeadModel::new(vs.root(), &config);
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
        .map(|input| Tensor::from_slice(&(input[..input.len() - 2])))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    // Forward pass
    let perm_mask = Tensor::zeros([1, 4, 4], (Kind::Float, device));
    let _ = perm_mask.narrow(2, 3, 1).fill_(1.0);

    let target_mapping = Tensor::zeros([1, 1, 4], (Kind::Float, device));
    let _ = target_mapping.narrow(2, 3, 1).fill_(1.0);
    let model_output = no_grad(|| {
        xlnet_model
            .forward_t(
                Some(&input_tensor),
                None,
                None,
                Some(perm_mask.as_ref()),
                Some(target_mapping.as_ref()),
                None,
                None,
                false,
            )
            .unwrap()
    });

    let index_1 = model_output.lm_logits.get(0).argmax(1, false);
    let word_1 = tokenizer.vocab().id_to_token(&index_1.int64_value(&[]));

    assert_eq!(word_1, "▁three".to_string());
    assert_eq!(model_output.lm_logits.size(), vec!(1, 1, 32000));
    assert!((model_output.lm_logits.double_value(&[0, 0, 139]) - -5.3240).abs() < 1e-4);
    Ok(())
}

#[test]
fn xlnet_generation_beam_search() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        XLNetConfigResources::XLNET_BASE_CASED,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        XLNetVocabResources::XLNET_BASE_CASED,
    ));
    let model_resource = Box::new(RemoteResource::from_pretrained(
        XLNetModelResources::XLNET_BASE_CASED,
    ));

    let generate_config = TextGenerationConfig {
        model_type: ModelType::XLNet,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: None,
        max_length: Some(32),
        do_sample: false,
        num_beams: 3,
        temperature: 1.0,
        num_return_sequences: 1,
        ..Default::default()
    };
    let model = TextGenerationModel::new(generate_config)?;

    let input_context = "Once upon a time,";
    let output = model.generate(&[input_context], None);

    assert_eq!(output.len(), 1);
    assert_eq!(
        output[0],
        " Once upon a time, there was a time when there was no one who could do magic. There was no one who could do magic. There was no one"
    );

    Ok(())
}

#[test]
fn xlnet_for_sequence_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        XLNetConfigResources::XLNET_BASE_CASED,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        XLNetVocabResources::XLNET_BASE_CASED,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer = XLNetTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let mut config = XLNetConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("Positive"));
    dummy_label_mapping.insert(1, String::from("Negative"));
    dummy_label_mapping.insert(3, String::from("Neutral"));
    config.id2label = Some(dummy_label_mapping);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let xlnet_model = XLNetForSequenceClassification::new(vs.root(), &config)?;

    //    Define input
    let input = ["Very positive sentence", "Second sentence input"];
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
        xlnet_model.forward_t(
            Some(input_tensor.as_ref()),
            None,
            None,
            None,
            None,
            None,
            None,
            false,
        )
    });

    assert_eq!(model_output.logits.size(), &[2, 3]);
    assert_eq!(
        config.n_layer as usize,
        model_output.all_hidden_states.unwrap().len()
    );
    assert_eq!(
        config.n_layer as usize,
        model_output.all_attentions.unwrap().len()
    );

    Ok(())
}

#[test]
fn xlnet_for_multiple_choice() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        XLNetConfigResources::XLNET_BASE_CASED,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        XLNetVocabResources::XLNET_BASE_CASED,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer = XLNetTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let config = XLNetConfig::from_file(config_path);
    let xlnet_model = XLNetForMultipleChoice::new(vs.root(), &config)?;

    //    Define input
    let prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced.";
    let inputs = ["Very positive sentence", "Second sentence input"];
    let tokenized_input = tokenizer.encode_pair_list(
        &inputs
            .iter()
            .map(|&inp| (prompt, inp))
            .collect::<Vec<(&str, &str)>>(),
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
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0)
        .to(device)
        .unsqueeze(0);

    //    Forward pass
    let model_output = no_grad(|| {
        xlnet_model.forward_t(
            Some(input_tensor.as_ref()),
            None,
            None,
            None,
            None,
            None,
            None,
            false,
        )
    });

    assert_eq!(model_output.logits.size(), &[1, 2]);

    Ok(())
}

#[test]
fn xlnet_for_token_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        XLNetConfigResources::XLNET_BASE_CASED,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        XLNetVocabResources::XLNET_BASE_CASED,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer = XLNetTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let mut config = XLNetConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("O"));
    dummy_label_mapping.insert(1, String::from("LOC"));
    dummy_label_mapping.insert(2, String::from("PER"));
    dummy_label_mapping.insert(3, String::from("ORG"));
    config.id2label = Some(dummy_label_mapping);
    let xlnet_model = XLNetForTokenClassification::new(vs.root(), &config)?;

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
    let model_output = no_grad(|| {
        xlnet_model.forward_t(
            Some(input_tensor.as_ref()),
            None,
            None,
            None,
            None,
            None,
            None,
            false,
        )
    });

    assert_eq!(model_output.logits.size(), &[2, 9, 4]);

    Ok(())
}

#[test]
fn xlnet_for_question_answering() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        XLNetConfigResources::XLNET_BASE_CASED,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        XLNetVocabResources::XLNET_BASE_CASED,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer = XLNetTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let config = XLNetConfig::from_file(config_path);
    let xlnet_model = XLNetForQuestionAnswering::new(vs.root(), &config)?;

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
    let model_output = no_grad(|| {
        xlnet_model.forward_t(
            Some(input_tensor.as_ref()),
            None,
            None,
            None,
            None,
            None,
            None,
            false,
        )
    });

    assert_eq!(model_output.start_logits.size(), &[1, 21]);
    assert_eq!(model_output.end_logits.size(), &[1, 21]);
    Ok(())
}
