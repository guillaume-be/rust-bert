use rust_bert::mobilebert::{
    MobileBertConfig, MobileBertConfigResources, MobileBertForMaskedLM,
    MobileBertForMultipleChoice, MobileBertForQuestionAnswering,
    MobileBertForSequenceClassification, MobileBertForTokenClassification,
    MobileBertModelResources, MobileBertVocabResources,
};
use rust_bert::pipelines::pos_tagging::POSModel;
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{BertTokenizer, MultiThreadedTokenizer, TruncationStrategy};
use rust_tokenizers::vocab::Vocab;
use std::collections::HashMap;
use std::convert::TryFrom;
use tch::{nn, no_grad, Device, Tensor};

#[test]
fn mobilebert_masked_model() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        MobileBertConfigResources::MOBILEBERT_UNCASED,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        MobileBertVocabResources::MOBILEBERT_UNCASED,
    ));
    let weights_resource = Box::new(RemoteResource::from_pretrained(
        MobileBertModelResources::MOBILEBERT_UNCASED,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    let tokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let mut config = MobileBertConfig::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let mobilebert_model = MobileBertForMaskedLM::new(vs.root(), &config);
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
            input.extend(vec![0; max_len - input.len()]);
            input
        })
        .map(|input| Tensor::from_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let model_output =
        no_grad(|| mobilebert_model.forward_t(Some(&input_tensor), None, None, None, None, false))?;

    //    Print masked tokens
    let index_1 = model_output.logits.get(0).get(4).argmax(0, false);
    let index_2 = model_output.logits.get(1).get(7).argmax(0, false);
    let word_1 = tokenizer.vocab().id_to_token(&index_1.int64_value(&[]));
    let word_2 = tokenizer.vocab().id_to_token(&index_2.int64_value(&[]));
    let score_1 = model_output
        .logits
        .get(0)
        .get(4)
        .double_value(&[i64::try_from(&index_1).unwrap()]);
    let score_2 = model_output
        .logits
        .get(1)
        .get(7)
        .double_value(&[i64::try_from(&index_2).unwrap()]);

    assert_eq!("thing", word_1); // Outputs "person" : "Looks like one [person] is missing"
    assert_eq!("sunny", word_2); // Outputs "sunny" : "It was a very nice and [sunny] day"
    assert!((score_1 - 10.0558).abs() < 1e-4);
    assert!((score_2 - 14.2708).abs() < 1e-4);

    assert_eq!(model_output.logits.size(), vec!(2, 10, config.vocab_size));
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
        vec!(2, 4, 10, 10)
    );
    assert_eq!(
        model_output.all_hidden_states.as_ref().unwrap()[0].size(),
        vec!(2, 10, 512)
    );

    Ok(())
}

#[test]
fn mobilebert_for_sequence_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        MobileBertConfigResources::MOBILEBERT_UNCASED,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        MobileBertVocabResources::MOBILEBERT_UNCASED,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let tokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let mut config = MobileBertConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("Positive"));
    dummy_label_mapping.insert(1, String::from("Negative"));
    dummy_label_mapping.insert(3, String::from("Neutral"));
    config.id2label = Some(dummy_label_mapping);
    let model = MobileBertForSequenceClassification::new(vs.root(), &config)?;

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
    let model_output =
        no_grad(|| model.forward_t(Some(input_tensor.as_ref()), None, None, None, None, false))?;

    assert_eq!(model_output.logits.size(), &[2, 3]);
    Ok(())
}

#[test]
fn mobilebert_for_multiple_choice() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        MobileBertConfigResources::MOBILEBERT_UNCASED,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        MobileBertVocabResources::MOBILEBERT_UNCASED,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let tokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let config = MobileBertConfig::from_file(config_path);
    let model = MobileBertForMultipleChoice::new(vs.root(), &config);

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
    let model_output =
        no_grad(|| model.forward_t(Some(input_tensor.as_ref()), None, None, None, None, false))?;

    assert_eq!(model_output.logits.size(), &[1, 2]);

    Ok(())
}

#[test]
fn mobilebert_for_token_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        MobileBertConfigResources::MOBILEBERT_UNCASED,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        MobileBertVocabResources::MOBILEBERT_UNCASED,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let tokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let mut config = MobileBertConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("O"));
    dummy_label_mapping.insert(1, String::from("LOC"));
    dummy_label_mapping.insert(2, String::from("PER"));
    dummy_label_mapping.insert(3, String::from("ORG"));
    config.id2label = Some(dummy_label_mapping);
    let model = MobileBertForTokenClassification::new(vs.root(), &config)?;

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
fn mobilebert_for_question_answering() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        MobileBertConfigResources::MOBILEBERT_UNCASED,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        MobileBertVocabResources::MOBILEBERT_UNCASED,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let tokenizer = BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let config = MobileBertConfig::from_file(config_path);
    let model = MobileBertForQuestionAnswering::new(vs.root(), &config);

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

#[test]
fn mobilebert_part_of_speech_tagging() -> anyhow::Result<()> {
    //    Set-up question answering model
    let pos_model = POSModel::new(Default::default())?;

    //    Define input
    let input = [
        "My name is Amélie. My email is amelie@somemail.com.",
        "A liter of milk costs 0.95 Euros!",
    ];

    let expected_outputs = [
        vec![
            ("My", 0.3144, "PRP"),
            ("name", 0.8918, "NN"),
            ("is", 0.8792, "VBZ"),
            ("Amélie", 0.9044, "NNP"),
            (".", 1.0, "."),
            ("My", 0.3244, "FW"),
            ("email", 0.9121, "NN"),
            ("is", 0.8167, "VBZ"),
            ("amelie", 0.9350, "NNP"),
            ("@", 0.7663, "IN"),
            ("somemail", 0.4503, "NNP"),
            (".", 0.8368, "NNP"),
            ("com", 0.9887, "NNP"),
            (".", 1.0, "."),
        ],
        vec![
            ("A", 0.9753, "DT"),
            ("liter", 0.9896, "NN"),
            ("of", 0.9988, "IN"),
            ("milk", 0.8592, "NN"),
            ("costs", 0.7448, "VBZ"),
            ("0", 0.9993, "CD"),
            (".", 0.9814, "CD"),
            ("95", 0.9998, "CD"),
            ("Euros", 0.8586, "NNS"),
            ("!", 1.0, "."),
        ],
    ];

    let answers = pos_model.predict(&input);

    assert_eq!(answers.len(), 2_usize);
    assert_eq!(answers[0].len(), expected_outputs[0].len());
    assert_eq!(answers[1].len(), expected_outputs[1].len());
    for (sequence_answer, expected_sequence_answer) in answers.iter().zip(expected_outputs.iter()) {
        assert_eq!(sequence_answer.len(), expected_sequence_answer.len());
        for (answer, expected_answer) in sequence_answer.iter().zip(expected_sequence_answer.iter())
        {
            assert_eq!(answer.word, expected_answer.0);
            assert_eq!(answer.label, expected_answer.2);
            assert!((answer.score - expected_answer.1).abs() < 1e-4);
        }
    }

    Ok(())
}
