use rust_bert::bert::BertConfig;
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::ner::NERModel;
use rust_bert::pipelines::question_answering::{
    QaInput, QuestionAnsweringConfig, QuestionAnsweringModel,
};
use rust_bert::pipelines::token_classification::TokenClassificationConfig;
use rust_bert::resources::{download_resource, RemoteResource, Resource};
use rust_bert::roberta::{
    RobertaConfigResources, RobertaForMaskedLM, RobertaForMultipleChoice,
    RobertaForQuestionAnswering, RobertaForSequenceClassification, RobertaForTokenClassification,
    RobertaMergesResources, RobertaModelResources, RobertaVocabResources,
};
use rust_bert::Config;
use rust_tokenizers::{RobertaTokenizer, Tokenizer, TruncationStrategy, Vocab};
use std::collections::HashMap;
use tch::{nn, no_grad, Device, Tensor};

#[test]
fn roberta_masked_lm() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaConfigResources::ROBERTA,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaVocabResources::ROBERTA,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaMergesResources::ROBERTA,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaModelResources::ROBERTA,
    ));
    let config_path = download_resource(&config_resource)?;
    let vocab_path = download_resource(&vocab_resource)?;
    let merges_path = download_resource(&merges_resource)?;
    let weights_path = download_resource(&weights_resource)?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: RobertaTokenizer = RobertaTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        true,
    );
    let config = BertConfig::from_file(config_path);
    let roberta_model = RobertaForMaskedLM::new(&vs.root(), &config);
    vs.load(weights_path)?;

    //    Define input
    let input = [
        "<pad> Looks like one thing is missing",
        "It\'s like comparing oranges to apples",
    ];
    let tokenized_input =
        tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
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
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let (output, _, _) = no_grad(|| {
        roberta_model.forward_t(
            Some(input_tensor),
            None,
            None,
            None,
            None,
            &None,
            &None,
            false,
        )
    });

    //    Print masked tokens
    let index_1 = output.get(0).get(4).argmax(0, false);
    let index_2 = output.get(1).get(5).argmax(0, false);
    let word_1 = tokenizer.vocab().id_to_token(&index_1.int64_value(&[]));
    let word_2 = tokenizer.vocab().id_to_token(&index_2.int64_value(&[]));

    assert_eq!("Ġsome", word_1); // Outputs "person" : "Looks like [some] thing is missing"
    assert_eq!("Ġapples", word_2); // Outputs "pear" : "It\'s like comparing [apples] to apples"

    Ok(())
}

#[test]
fn roberta_for_sequence_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaConfigResources::ROBERTA,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaVocabResources::ROBERTA,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaMergesResources::ROBERTA,
    ));
    let config_path = download_resource(&config_resource)?;
    let vocab_path = download_resource(&vocab_resource)?;
    let merges_path = download_resource(&merges_resource)?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: RobertaTokenizer = RobertaTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        true,
    );
    let mut config = BertConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("Positive"));
    dummy_label_mapping.insert(1, String::from("Negative"));
    dummy_label_mapping.insert(3, String::from("Neutral"));
    config.id2label = Some(dummy_label_mapping);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let roberta_model = RobertaForSequenceClassification::new(&vs.root(), &config);

    //    Define input
    let input = [
        "Looks like one thing is missing",
        "It\'s like comparing oranges to apples",
    ];
    let tokenized_input =
        tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
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
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let (output, all_hidden_states, all_attentions) =
        no_grad(|| roberta_model.forward_t(Some(input_tensor), None, None, None, None, false));

    assert_eq!(output.size(), &[2, 3]);
    assert_eq!(
        config.num_hidden_layers as usize,
        all_hidden_states.unwrap().len()
    );
    assert_eq!(
        config.num_hidden_layers as usize,
        all_attentions.unwrap().len()
    );

    Ok(())
}

#[test]
fn roberta_for_multiple_choice() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaConfigResources::ROBERTA,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaVocabResources::ROBERTA,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaMergesResources::ROBERTA,
    ));
    let config_path = download_resource(&config_resource)?;
    let vocab_path = download_resource(&vocab_resource)?;
    let merges_path = download_resource(&merges_resource)?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: RobertaTokenizer = RobertaTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        true,
    );
    let mut config = BertConfig::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let roberta_model = RobertaForMultipleChoice::new(&vs.root(), &config);

    //    Define input
    let input = [
        "Looks like one thing is missing",
        "It\'s like comparing oranges to apples",
    ];
    let tokenized_input =
        tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
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
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0)
        .to(device)
        .unsqueeze(0);

    //    Forward pass
    let (output, all_hidden_states, all_attentions) =
        no_grad(|| roberta_model.forward_t(input_tensor, None, None, None, false));

    assert_eq!(output.size(), &[1, 2]);
    assert_eq!(
        config.num_hidden_layers as usize,
        all_hidden_states.unwrap().len()
    );
    assert_eq!(
        config.num_hidden_layers as usize,
        all_attentions.unwrap().len()
    );

    Ok(())
}

#[test]
fn roberta_for_token_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaConfigResources::ROBERTA,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaVocabResources::ROBERTA,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaMergesResources::ROBERTA,
    ));
    let config_path = download_resource(&config_resource)?;
    let vocab_path = download_resource(&vocab_resource)?;
    let merges_path = download_resource(&merges_resource)?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: RobertaTokenizer = RobertaTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        true,
    );
    let mut config = BertConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("O"));
    dummy_label_mapping.insert(1, String::from("LOC"));
    dummy_label_mapping.insert(2, String::from("PER"));
    dummy_label_mapping.insert(3, String::from("ORG"));
    config.id2label = Some(dummy_label_mapping);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let roberta_model = RobertaForTokenClassification::new(&vs.root(), &config);

    //    Define input
    let input = [
        "Looks like one thing is missing",
        "It\'s like comparing oranges to apples",
    ];
    let tokenized_input =
        tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
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
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let (output, all_hidden_states, all_attentions) =
        no_grad(|| roberta_model.forward_t(Some(input_tensor), None, None, None, None, false));

    assert_eq!(output.size(), &[2, 9, 4]);
    assert_eq!(
        config.num_hidden_layers as usize,
        all_hidden_states.unwrap().len()
    );
    assert_eq!(
        config.num_hidden_layers as usize,
        all_attentions.unwrap().len()
    );

    Ok(())
}

#[test]
fn roberta_for_question_answering() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaConfigResources::ROBERTA,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaVocabResources::ROBERTA,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaMergesResources::ROBERTA,
    ));
    let config_path = download_resource(&config_resource)?;
    let vocab_path = download_resource(&vocab_resource)?;
    let merges_path = download_resource(&merges_resource)?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer: RobertaTokenizer = RobertaTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        true,
    );
    let mut config = BertConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("Positive"));
    dummy_label_mapping.insert(1, String::from("Negative"));
    dummy_label_mapping.insert(3, String::from("Neutral"));
    config.id2label = Some(dummy_label_mapping);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let roberta_model = RobertaForQuestionAnswering::new(&vs.root(), &config);

    //    Define input
    let input = [
        "Looks like one thing is missing",
        "It\'s like comparing oranges to apples",
    ];
    let tokenized_input =
        tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
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
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let (start_scores, end_scores, all_hidden_states, all_attentions) =
        no_grad(|| roberta_model.forward_t(Some(input_tensor), None, None, None, None, false));

    assert_eq!(start_scores.size(), &[2, 9]);
    assert_eq!(end_scores.size(), &[2, 9]);
    assert_eq!(
        config.num_hidden_layers as usize,
        all_hidden_states.unwrap().len()
    );
    assert_eq!(
        config.num_hidden_layers as usize,
        all_attentions.unwrap().len()
    );

    Ok(())
}

#[test]
fn roberta_question_answering() -> anyhow::Result<()> {
    //    Set-up question answering model
    let config = QuestionAnsweringConfig::new(
        ModelType::Roberta,
        Resource::Remote(RemoteResource::from_pretrained(
            RobertaModelResources::ROBERTA_QA,
        )),
        Resource::Remote(RemoteResource::from_pretrained(
            RobertaConfigResources::ROBERTA_QA,
        )),
        Resource::Remote(RemoteResource::from_pretrained(
            RobertaVocabResources::ROBERTA_QA,
        )),
        Some(Resource::Remote(RemoteResource::from_pretrained(
            RobertaMergesResources::ROBERTA_QA,
        ))), //merges resource only relevant with ModelType::Roberta
        true, //lowercase
    );

    let qa_model = QuestionAnsweringModel::new(config)?;

    //    Define input
    let question = String::from("Where does Amy live ?");
    let context = String::from("Amy lives in Amsterdam");
    let qa_input = QaInput { question, context };

    let answers = qa_model.predict(&vec![qa_input], 1, 32);

    assert_eq!(answers.len(), 1 as usize);
    assert_eq!(answers[0].len(), 1 as usize);
    assert_eq!(answers[0][0].start, 13);
    assert_eq!(answers[0][0].end, 21);
    assert!((answers[0][0].score - 0.7354).abs() < 1e-4);
    assert_eq!(answers[0][0].answer, "Amsterdam");

    Ok(())
}

#[test]
fn xlm_roberta_german_ner() -> anyhow::Result<()> {
    //    Set-up question answering model
    let ner_config = TokenClassificationConfig {
        model_type: ModelType::XLMRoberta,
        model_resource: Resource::Remote(RemoteResource::from_pretrained(
            RobertaModelResources::XLM_ROBERTA_NER_DE,
        )),
        config_resource: Resource::Remote(RemoteResource::from_pretrained(
            RobertaConfigResources::XLM_ROBERTA_NER_DE,
        )),
        vocab_resource: Resource::Remote(RemoteResource::from_pretrained(
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

    assert_eq!(output.len(), 4);

    assert_eq!(output[0].word, " Amélie");
    assert!((output[0].score - 0.9983).abs() < 1e-4);
    assert_eq!(output[0].label, "I-PER");

    assert_eq!(output[1].word, " Москва");
    assert!((output[1].score - 0.9999).abs() < 1e-4);
    assert_eq!(output[1].label, "I-LOC");

    assert_eq!(output[2].word, "Chongqing");
    assert!((output[2].score - 0.9997).abs() < 1e-4);
    assert_eq!(output[2].label, "I-LOC");

    assert_eq!(output[3].word, " China");
    assert!((output[3].score - 0.9999).abs() < 1e-4);
    assert_eq!(output[3].label, "I-LOC");

    Ok(())
}
