extern crate anyhow;
extern crate dirs;

use rust_bert::bert::{
    BertConfig, BertConfigResources, BertForMaskedLM, BertForMultipleChoice,
    BertForQuestionAnswering, BertForSequenceClassification, BertForTokenClassification,
    BertModelResources, BertVocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::masked_language::{MaskedLanguageConfig, MaskedLanguageModel};
use rust_bert::pipelines::ner::NERModel;
use rust_bert::pipelines::question_answering::{
    QaInput, QuestionAnsweringConfig, QuestionAnsweringModel,
};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{BertTokenizer, MultiThreadedTokenizer, TruncationStrategy};
use rust_tokenizers::vocab::Vocab;
use std::collections::HashMap;
use tch::{nn, no_grad, Device, Tensor};

#[test]
fn bert_masked_lm() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = RemoteResource::from_pretrained(BertConfigResources::BERT);
    let vocab_resource = RemoteResource::from_pretrained(BertVocabResources::BERT);
    let weights_resource = RemoteResource::from_pretrained(BertModelResources::BERT);
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer =
        BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let config = BertConfig::from_file(config_path);
    let bert_model = BertForMaskedLM::new(vs.root(), &config);
    vs.load(weights_path)?;

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
    let mut tokenized_input = tokenized_input
        .iter()
        .map(|input| input.token_ids.clone())
        .map(|mut input| {
            input.extend(vec![0; max_len - input.len()]);
            input
        })
        .collect::<Vec<_>>();

    //    Masking the token [thing] of sentence 1 and [oranges] of sentence 2
    tokenized_input[0][4] = 103;
    tokenized_input[1][6] = 103;
    let tokenized_input = tokenized_input
        .iter()
        .map(|input| Tensor::from_slice(input))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let model_output = no_grad(|| {
        bert_model.forward_t(
            Some(&input_tensor),
            None,
            None,
            None,
            None,
            None,
            None,
            false,
        )
    });

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

    assert_eq!("person", word_1); // Outputs "person" : "Looks like one [person] is missing"
    assert_eq!("orange", word_2); // Outputs "pear" : "It\'s like comparing [pear] to apples"

    Ok(())
}

#[test]
fn bert_masked_lm_pipeline() -> anyhow::Result<()> {
    //    Set-up model
    let config = MaskedLanguageConfig::new(
        ModelType::Bert,
        RemoteResource::from_pretrained(BertModelResources::BERT),
        RemoteResource::from_pretrained(BertConfigResources::BERT),
        RemoteResource::from_pretrained(BertVocabResources::BERT),
        None,
        true,
        None,
        None,
        Some(String::from("<mask>")),
    );

    let mask_language_model = MaskedLanguageModel::new(config)?;
    //    Define input
    let input = [
        "Hello I am a <mask> student",
        "Paris is the <mask> of France. It is <mask> in Europe.",
    ];

    //    Run model
    let output = mask_language_model.predict(input)?;

    assert_eq!(output.len(), 2);
    assert_eq!(output[0].len(), 1);
    assert_eq!(output[0][0].id, 2267);
    assert_eq!(output[0][0].text, "college");
    assert!((output[0][0].score - 8.0919).abs() < 1e-4);
    assert_eq!(output[1].len(), 2);
    assert_eq!(output[1][0].id, 3007);
    assert_eq!(output[1][0].text, "capital");
    assert!((output[1][0].score - 16.7249).abs() < 1e-4);
    assert_eq!(output[1][1].id, 2284);
    assert_eq!(output[1][1].text, "located");
    assert!((output[1][1].score - 9.0452).abs() < 1e-4);
    Ok(())
}

#[test]
fn bert_for_sequence_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = RemoteResource::from_pretrained(BertConfigResources::BERT);
    let vocab_resource = RemoteResource::from_pretrained(BertVocabResources::BERT);
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer =
        BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let mut config = BertConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("Positive"));
    dummy_label_mapping.insert(1, String::from("Negative"));
    dummy_label_mapping.insert(3, String::from("Neutral"));
    config.id2label = Some(dummy_label_mapping);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let bert_model = BertForSequenceClassification::new(vs.root(), &config)?;

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
        no_grad(|| bert_model.forward_t(Some(&input_tensor), None, None, None, None, false));

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
fn bert_for_multiple_choice() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = RemoteResource::from_pretrained(BertConfigResources::BERT);
    let vocab_resource = RemoteResource::from_pretrained(BertVocabResources::BERT);
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer =
        BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let mut config = BertConfig::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let bert_model = BertForMultipleChoice::new(vs.root(), &config);

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
    let model_output = no_grad(|| bert_model.forward_t(&input_tensor, None, None, None, false));

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
fn bert_for_token_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = RemoteResource::from_pretrained(BertConfigResources::BERT);
    let vocab_resource = RemoteResource::from_pretrained(BertVocabResources::BERT);
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer =
        BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let mut config = BertConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("O"));
    dummy_label_mapping.insert(1, String::from("LOC"));
    dummy_label_mapping.insert(2, String::from("PER"));
    dummy_label_mapping.insert(3, String::from("ORG"));
    config.id2label = Some(dummy_label_mapping);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let bert_model = BertForTokenClassification::new(vs.root(), &config)?;

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
        no_grad(|| bert_model.forward_t(Some(&input_tensor), None, None, None, None, false));

    assert_eq!(model_output.logits.size(), &[2, 11, 4]);
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
fn bert_for_question_answering() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = RemoteResource::from_pretrained(BertConfigResources::BERT);
    let vocab_resource = RemoteResource::from_pretrained(BertVocabResources::BERT);
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer =
        BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let mut config = BertConfig::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let bert_model = BertForQuestionAnswering::new(vs.root(), &config);

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
        no_grad(|| bert_model.forward_t(Some(&input_tensor), None, None, None, None, false));

    assert_eq!(model_output.start_logits.size(), &[2, 11]);
    assert_eq!(model_output.end_logits.size(), &[2, 11]);
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
fn bert_pre_trained_ner() -> anyhow::Result<()> {
    //    Set-up model
    let ner_model = NERModel::new(Default::default())?;

    //    Define input
    let input = [
        "My name is Amy. I live in Paris.",
        "Paris is a city in France.",
    ];

    //    Run model
    let output = ner_model.predict(&input);

    assert_eq!(output.len(), 2);
    assert_eq!(output[0].len(), 2);
    assert_eq!(output[1].len(), 2);

    assert_eq!(output[0][0].word, "Amy");
    assert!((output[0][0].score - 0.9986).abs() < 1e-4);
    assert_eq!(output[0][0].label, "I-PER");

    assert_eq!(output[0][1].word, "Paris");
    assert!((output[0][1].score - 0.9986).abs() < 1e-4);
    assert_eq!(output[0][1].label, "I-LOC");

    assert_eq!(output[1][0].word, "Paris");
    assert!((output[1][0].score - 0.9981).abs() < 1e-4);
    assert_eq!(output[1][0].label, "I-LOC");

    assert_eq!(output[1][1].word, "France");
    assert!((output[1][1].score - 0.9984).abs() < 1e-4);
    assert_eq!(output[1][1].label, "I-LOC");

    Ok(())
}

#[test]
fn bert_pre_trained_ner_full_entities() -> anyhow::Result<()> {
    //    Set-up model
    let ner_model = NERModel::new(Default::default())?;

    //    Define input
    let input = ["Asked John Smith about Acme Corp", "Let's go to New York!"];

    //    Run model
    let output = ner_model.predict_full_entities(&input);

    assert_eq!(output.len(), 2);

    assert_eq!(output[0][0].word, "John Smith");
    assert!((output[0][0].score - 0.9872).abs() < 1e-4);
    assert_eq!(output[0][0].label, "PER");

    assert_eq!(output[0][1].word, "Acme Corp");
    assert!((output[0][1].score - 0.9622).abs() < 1e-4);
    assert_eq!(output[0][1].label, "ORG");

    assert_eq!(output[1][0].word, "New York");
    assert!((output[1][0].score - 0.9991).abs() < 1e-4);
    assert_eq!(output[1][0].label, "LOC");

    Ok(())
}

#[test]
fn bert_question_answering() -> anyhow::Result<()> {
    //    Set-up question answering model
    let config = QuestionAnsweringConfig {
        model_type: ModelType::Bert,
        model_resource: Box::new(RemoteResource::from_pretrained(BertModelResources::BERT_QA)),
        config_resource: Box::new(RemoteResource::from_pretrained(
            BertConfigResources::BERT_QA,
        )),
        vocab_resource: Box::new(RemoteResource::from_pretrained(BertVocabResources::BERT_QA)),
        lower_case: false,
        strip_accents: Some(false),
        add_prefix_space: None,
        device: Device::Cpu,
        ..Default::default()
    };

    let qa_model = QuestionAnsweringModel::new(config)?;

    //    Define input
    let question = String::from("Where does Amy live ?");
    let context = String::from("Amy lives in Amsterdam");
    let qa_input = QaInput { question, context };

    let answers = qa_model.predict(&[qa_input], 1, 32);

    assert_eq!(answers.len(), 1usize);
    assert_eq!(answers[0].len(), 1usize);
    assert_eq!(answers[0][0].start, 13);
    assert_eq!(answers[0][0].end, 22);
    assert!((answers[0][0].score - 0.9806).abs() < 1e-4);
    assert_eq!(answers[0][0].answer, "Amsterdam");

    Ok(())
}
