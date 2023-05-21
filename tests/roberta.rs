use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::ner::NERModel;
use rust_bert::pipelines::question_answering::{
    QaInput, QuestionAnsweringConfig, QuestionAnsweringModel,
};
use rust_bert::pipelines::token_classification::TokenClassificationConfig;
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::roberta::{
    RobertaConfig, RobertaConfigResources, RobertaForMaskedLM, RobertaForMultipleChoice,
    RobertaForSequenceClassification, RobertaForTokenClassification, RobertaMergesResources,
    RobertaModelResources, RobertaVocabResources,
};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{RobertaTokenizer, Tokenizer, TruncationStrategy};
use rust_tokenizers::vocab::Vocab;
use std::collections::HashMap;
use tch::{nn, no_grad, Device, Tensor};

#[test]
fn roberta_masked_lm() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource =
        RemoteResource::from_pretrained(RobertaConfigResources::DISTILROBERTA_BASE);
    let vocab_resource = RemoteResource::from_pretrained(RobertaVocabResources::DISTILROBERTA_BASE);
    let merges_resource =
        RemoteResource::from_pretrained(RobertaMergesResources::DISTILROBERTA_BASE);
    let weights_resource =
        RemoteResource::from_pretrained(RobertaModelResources::DISTILROBERTA_BASE);
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let merges_path = merges_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: RobertaTokenizer = RobertaTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        true,
        false,
    )?;
    let config = RobertaConfig::from_file(config_path);
    let roberta_model = RobertaForMaskedLM::new(vs.root(), &config);
    vs.load(weights_path)?;

    //    Define input
    let input = [
        "<pad> Looks like one thing is missing",
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
    tokenized_input[1][5] = 103;
    let tokenized_input = tokenized_input
        .iter()
        .map(|input| Tensor::from_slice(input))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let model_output = no_grad(|| {
        roberta_model.forward_t(
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
        .get(5)
        .argmax(0, false);
    let word_1 = tokenizer.vocab().id_to_token(&index_1.int64_value(&[]));
    let word_2 = tokenizer.vocab().id_to_token(&index_2.int64_value(&[]));

    assert_eq!("Ġsome", word_1); // Outputs "person" : "Looks like [some] thing is missing"
    assert_eq!("Ġsome", word_2); // Outputs "pear" : "It\'s like comparing [apples] to apples"

    Ok(())
}

#[test]
fn roberta_for_sequence_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource =
        RemoteResource::from_pretrained(RobertaConfigResources::DISTILROBERTA_BASE);
    let vocab_resource = RemoteResource::from_pretrained(RobertaVocabResources::DISTILROBERTA_BASE);
    let merges_resource =
        RemoteResource::from_pretrained(RobertaMergesResources::DISTILROBERTA_BASE);
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let merges_path = merges_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: RobertaTokenizer = RobertaTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        true,
        false,
    )?;
    let mut config = RobertaConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("Positive"));
    dummy_label_mapping.insert(1, String::from("Negative"));
    dummy_label_mapping.insert(3, String::from("Neutral"));
    config.id2label = Some(dummy_label_mapping);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let roberta_model = RobertaForSequenceClassification::new(vs.root(), &config)?;

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
        no_grad(|| roberta_model.forward_t(Some(&input_tensor), None, None, None, None, false));

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
fn roberta_for_multiple_choice() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource =
        RemoteResource::from_pretrained(RobertaConfigResources::DISTILROBERTA_BASE);
    let vocab_resource = RemoteResource::from_pretrained(RobertaVocabResources::DISTILROBERTA_BASE);
    let merges_resource =
        RemoteResource::from_pretrained(RobertaMergesResources::DISTILROBERTA_BASE);
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let merges_path = merges_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: RobertaTokenizer = RobertaTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        true,
        false,
    )?;
    let mut config = RobertaConfig::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let roberta_model = RobertaForMultipleChoice::new(vs.root(), &config);

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
    let model_output = no_grad(|| roberta_model.forward_t(&input_tensor, None, None, None, false));

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
fn roberta_for_token_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource =
        RemoteResource::from_pretrained(RobertaConfigResources::DISTILROBERTA_BASE);
    let vocab_resource = RemoteResource::from_pretrained(RobertaVocabResources::DISTILROBERTA_BASE);
    let merges_resource =
        RemoteResource::from_pretrained(RobertaMergesResources::DISTILROBERTA_BASE);
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let merges_path = merges_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: RobertaTokenizer = RobertaTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        true,
        false,
    )?;
    let mut config = RobertaConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("O"));
    dummy_label_mapping.insert(1, String::from("LOC"));
    dummy_label_mapping.insert(2, String::from("PER"));
    dummy_label_mapping.insert(3, String::from("ORG"));
    config.id2label = Some(dummy_label_mapping);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let roberta_model = RobertaForTokenClassification::new(vs.root(), &config)?;

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
        no_grad(|| roberta_model.forward_t(Some(&input_tensor), None, None, None, None, false));

    assert_eq!(model_output.logits.size(), &[2, 9, 4]);
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
fn roberta_question_answering() -> anyhow::Result<()> {
    //    Set-up question answering model
    let config = QuestionAnsweringConfig::new(
        ModelType::Roberta,
        RemoteResource::from_pretrained(RobertaModelResources::ROBERTA_QA),
        RemoteResource::from_pretrained(RobertaConfigResources::ROBERTA_QA),
        RemoteResource::from_pretrained(RobertaVocabResources::ROBERTA_QA),
        Some(RemoteResource::from_pretrained(
            RobertaMergesResources::ROBERTA_QA,
        )),
        false,
        None,
        false,
    );

    let qa_model = QuestionAnsweringModel::new(config)?;

    //    Define input
    let question = String::from("Where does Amy live ?");
    let context = String::from("Amy lives in Amsterdam");
    let qa_input = QaInput { question, context };

    let answers = qa_model.predict(&[qa_input], 1, 32);

    assert_eq!(answers.len(), 1usize);
    assert_eq!(answers[0].len(), 1usize);
    assert_eq!(answers[0][0].start, 12);
    assert_eq!(answers[0][0].end, 22);
    assert!((answers[0][0].score - 0.9997).abs() < 1e-4);
    assert_eq!(answers[0][0].answer, " Amsterdam");

    Ok(())
}

#[test]
fn xlm_roberta_german_ner() -> anyhow::Result<()> {
    //    Set-up question answering model
    let ner_config = TokenClassificationConfig {
        model_type: ModelType::XLMRoberta,
        model_resource: Box::new(RemoteResource::from_pretrained(
            RobertaModelResources::XLM_ROBERTA_NER_DE,
        )),
        config_resource: Box::new(RemoteResource::from_pretrained(
            RobertaConfigResources::XLM_ROBERTA_NER_DE,
        )),
        vocab_resource: Box::new(RemoteResource::from_pretrained(
            RobertaVocabResources::XLM_ROBERTA_NER_DE,
        )),
        lower_case: false,
        device: Device::cuda_if_available(),
        ..Default::default()
    };

    let ner_model = NERModel::new(ner_config)?;

    //    Define input
    let input = [
        "Mein Name ist Amélie. Ich lebe in Москва.",
        "Chongqing ist eine Stadt in China.",
    ];

    let output = ner_model.predict(&input);

    assert_eq!(output.len(), 2);
    assert_eq!(output[0].len(), 2);
    assert_eq!(output[1].len(), 2);

    assert_eq!(output[0][0].word, " Amélie");
    assert!((output[0][0].score - 0.9983).abs() < 1e-4);
    assert_eq!(output[0][0].label, "I-PER");

    assert_eq!(output[0][1].word, " Москва");
    assert!((output[0][1].score - 0.9999).abs() < 1e-4);
    assert_eq!(output[0][1].label, "I-LOC");

    assert_eq!(output[1][0].word, "Chongqing");
    assert!((output[1][0].score - 0.9997).abs() < 1e-4);
    assert_eq!(output[1][0].label, "I-LOC");

    assert_eq!(output[1][1].word, " China");
    assert!((output[1][1].score - 0.9999).abs() < 1e-4);
    assert_eq!(output[1][1].label, "I-LOC");

    Ok(())
}
