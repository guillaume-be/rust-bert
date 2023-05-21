use rust_bert::distilbert::{
    DistilBertConfig, DistilBertConfigResources, DistilBertForQuestionAnswering,
    DistilBertForTokenClassification, DistilBertModelMaskedLM, DistilBertModelResources,
    DistilBertVocabResources,
};
use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
use rust_bert::pipelines::sentiment::{SentimentModel, SentimentPolarity};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{BertTokenizer, MultiThreadedTokenizer, TruncationStrategy};
use rust_tokenizers::vocab::Vocab;
use std::collections::HashMap;
use tch::{nn, no_grad, Device, Tensor};

extern crate anyhow;

#[test]
fn distilbert_sentiment_classifier() -> anyhow::Result<()> {
    //    Set-up classifier
    let sentiment_classifier = SentimentModel::new(Default::default())?;

    //    Get sentiments
    let input = [
        "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
        "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
        "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
    ];

    let output = sentiment_classifier.predict(input);

    assert_eq!(output.len(), 3usize);
    assert_eq!(output[0].polarity, SentimentPolarity::Positive);
    assert!((output[0].score - 0.9981).abs() < 1e-4);
    assert_eq!(output[1].polarity, SentimentPolarity::Negative);
    assert!((output[1].score - 0.9927).abs() < 1e-4);
    assert_eq!(output[2].polarity, SentimentPolarity::Positive);
    assert!((output[2].score - 0.9997).abs() < 1e-4);

    Ok(())
}

#[test]
fn distilbert_masked_lm() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        DistilBertConfigResources::DISTIL_BERT,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        DistilBertVocabResources::DISTIL_BERT,
    ));
    let weights_resource = Box::new(RemoteResource::from_pretrained(
        DistilBertModelResources::DISTIL_BERT,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer =
        BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let config = DistilBertConfig::from_file(config_path);
    let distil_bert_model = DistilBertModelMaskedLM::new(vs.root(), &config);
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
        distil_bert_model
            .forward_t(Some(&input_tensor), None, None, false)
            .unwrap()
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
    assert_eq!("pear", word_2); // Outputs "pear" : "It\'s like comparing [pear] to apples"

    Ok(())
}

#[test]
fn distilbert_for_question_answering() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        DistilBertConfigResources::DISTIL_BERT_SQUAD,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        DistilBertVocabResources::DISTIL_BERT_SQUAD,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer =
        BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let mut config = DistilBertConfig::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let distil_bert_model = DistilBertForQuestionAnswering::new(vs.root(), &config);

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
        distil_bert_model
            .forward_t(Some(&input_tensor), None, None, false)
            .unwrap()
    });

    assert_eq!(model_output.start_logits.size(), &[2, 11]);
    assert_eq!(model_output.end_logits.size(), &[2, 11]);
    assert_eq!(
        config.n_layers as usize,
        model_output.all_hidden_states.unwrap().len()
    );
    assert_eq!(
        config.n_layers as usize,
        model_output.all_attentions.unwrap().len()
    );

    Ok(())
}

#[test]
fn distilbert_for_token_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        DistilBertConfigResources::DISTIL_BERT,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        DistilBertVocabResources::DISTIL_BERT,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer =
        BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let mut config = DistilBertConfig::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("O"));
    dummy_label_mapping.insert(1, String::from("LOC"));
    dummy_label_mapping.insert(2, String::from("PER"));
    dummy_label_mapping.insert(3, String::from("ORG"));
    config.id2label = Some(dummy_label_mapping);
    let distil_bert_model = DistilBertForTokenClassification::new(vs.root(), &config)?;

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
        distil_bert_model
            .forward_t(Some(&input_tensor), None, None, false)
            .unwrap()
    });

    assert_eq!(model_output.logits.size(), &[2, 11, 4]);
    assert_eq!(
        config.n_layers as usize,
        model_output.all_hidden_states.unwrap().len()
    );
    assert_eq!(
        config.n_layers as usize,
        model_output.all_attentions.unwrap().len()
    );

    Ok(())
}

#[test]
fn distilbert_question_answering() -> anyhow::Result<()> {
    //    Set-up question answering model
    let qa_model = QuestionAnsweringModel::new(Default::default())?;

    //    Define input
    let question = String::from("Where does Amy live ?");
    let context = String::from("Amy lives in Amsterdam");
    let qa_input = QaInput { question, context };

    let answers = qa_model.predict(&[qa_input], 1, 32);

    assert_eq!(answers.len(), 1usize);
    assert_eq!(answers[0].len(), 1usize);
    assert_eq!(answers[0][0].start, 13);
    assert_eq!(answers[0][0].end, 22);
    assert!((answers[0][0].score - 0.9978).abs() < 1e-4);
    assert_eq!(answers[0][0].answer, "Amsterdam");

    Ok(())
}
