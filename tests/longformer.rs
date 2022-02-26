extern crate anyhow;
extern crate dirs;

use rust_bert::longformer::{
    LongformerConfig, LongformerConfigResources, LongformerForMaskedLM,
    LongformerForMultipleChoice, LongformerForSequenceClassification,
    LongformerForTokenClassification, LongformerMergesResources, LongformerModelResources,
    LongformerVocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::question_answering::{
    QaInput, QuestionAnsweringConfig, QuestionAnsweringModel,
};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{MultiThreadedTokenizer, RobertaTokenizer, TruncationStrategy};
use rust_tokenizers::vocab::{RobertaVocab, Vocab};
use std::collections::HashMap;
use tch::{nn, no_grad, Device, Tensor};

#[test]
fn longformer_masked_lm() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource =
        RemoteResource::from_pretrained(LongformerConfigResources::LONGFORMER_BASE_4096);
    let vocab_resource =
        RemoteResource::from_pretrained(LongformerVocabResources::LONGFORMER_BASE_4096);
    let merges_resource =
        RemoteResource::from_pretrained(LongformerMergesResources::LONGFORMER_BASE_4096);
    let weights_resource =
        RemoteResource::from_pretrained(LongformerModelResources::LONGFORMER_BASE_4096);
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let merges_path = merges_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    let tokenizer = RobertaTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        false,
        false,
    )?;
    let mut config = LongformerConfig::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let model = LongformerForMaskedLM::new(vs.root(), &config);

    vs.load(weights_path)?;

    //    Define input
    let input = [
        "Looks like one <mask> is missing",
        "It was a very nice and <mask> day",
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
            input.extend(vec![
                tokenizer.vocab().token_to_id(RobertaVocab::pad_value());
                max_len - input.len()
            ]);
            input
        })
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);
    let mut global_attention_mask_vector = vec![0; max_len];
    global_attention_mask_vector[0] = 1;
    let global_attention_mask = Tensor::of_slice(global_attention_mask_vector.as_slice());
    let global_attention_mask = Tensor::stack(
        vec![&global_attention_mask; tokenized_input.len()].as_slice(),
        0,
    )
    .to(device);

    //    Forward pass
    let model_output = no_grad(|| {
        model.forward_t(
            Some(&input_tensor),
            None,
            Some(global_attention_mask.as_ref()),
            None,
            None,
            None,
            false,
        )
    })?;

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
    let score_1 = model_output
        .prediction_scores
        .get(0)
        .get(4)
        .double_value(&[i64::from(&index_1)]);
    let score_2 = model_output
        .prediction_scores
        .get(1)
        .get(7)
        .double_value(&[i64::from(&index_2)]);

    assert_eq!("Ġeye", word_1); // Outputs "person" : "Looks like one [eye] is missing"
    assert_eq!("Ġsunny", word_2); // Outputs "pear" : "It was a nice and [sunny] day"

    assert!((score_1 - 11.7605).abs() < 1e-4);
    assert!((score_2 - 17.0088).abs() < 1e-4);

    assert_eq!(
        model_output.prediction_scores.size(),
        vec!(2, 10, config.vocab_size)
    );
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
        vec!(
            2,
            config.num_attention_heads,
            *config.attention_window.iter().max().unwrap(),
            1 + *config.attention_window.iter().max().unwrap() + 1
        )
    );
    assert_eq!(
        model_output.all_global_attentions.as_ref().unwrap()[0].size(),
        vec!(
            2,
            config.num_attention_heads,
            *config.attention_window.iter().max().unwrap(),
            1
        )
    );
    assert_eq!(
        model_output.all_hidden_states.as_ref().unwrap()[0].size(),
        vec!(
            2,
            *config.attention_window.iter().max().unwrap(),
            config.hidden_size
        )
    );
    Ok(())
}
#[test]
fn longformer_for_sequence_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource =
        RemoteResource::from_pretrained(LongformerConfigResources::LONGFORMER_BASE_4096);
    let vocab_resource =
        RemoteResource::from_pretrained(LongformerVocabResources::LONGFORMER_BASE_4096);
    let merges_resource =
        RemoteResource::from_pretrained(LongformerMergesResources::LONGFORMER_BASE_4096);
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let merges_path = merges_resource.get_local_path()?;

    //    Set-up model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let tokenizer = RobertaTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        false,
        false,
    )?;
    let mut config = LongformerConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("Positive"));
    dummy_label_mapping.insert(1, String::from("Negative"));
    dummy_label_mapping.insert(3, String::from("Neutral"));
    config.id2label = Some(dummy_label_mapping);
    let model = LongformerForSequenceClassification::new(&vs.root(), &config);

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
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let model_output = no_grad(|| {
        model.forward_t(
            Some(input_tensor.as_ref()),
            None,
            None,
            None,
            None,
            None,
            false,
        )
    })?;

    assert_eq!(model_output.logits.size(), &[2, 3]);
    Ok(())
}

#[test]
fn longformer_for_multiple_choice() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource =
        RemoteResource::from_pretrained(LongformerConfigResources::LONGFORMER_BASE_4096);
    let vocab_resource =
        RemoteResource::from_pretrained(LongformerVocabResources::LONGFORMER_BASE_4096);
    let merges_resource =
        RemoteResource::from_pretrained(LongformerMergesResources::LONGFORMER_BASE_4096);
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let merges_path = merges_resource.get_local_path()?;

    //    Set-up model
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let tokenizer = RobertaTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        false,
        false,
    )?;
    let config = LongformerConfig::from_file(config_path);
    let model = LongformerForMultipleChoice::new(&vs.root(), &config);

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
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0)
        .to(device)
        .unsqueeze(0);

    //    Forward pass
    let model_output = no_grad(|| {
        model.forward_t(
            Some(input_tensor.as_ref()),
            None,
            None,
            None,
            None,
            None,
            false,
        )
    })?;

    assert_eq!(model_output.logits.size(), &[1, 2]);

    Ok(())
}

#[test]
fn mobilebert_for_token_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource =
        RemoteResource::from_pretrained(LongformerConfigResources::LONGFORMER_BASE_4096);
    let vocab_resource =
        RemoteResource::from_pretrained(LongformerVocabResources::LONGFORMER_BASE_4096);
    let merges_resource =
        RemoteResource::from_pretrained(LongformerMergesResources::LONGFORMER_BASE_4096);
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let merges_path = merges_resource.get_local_path()?;

    //    Set-up model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let tokenizer = RobertaTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        false,
        false,
    )?;
    let mut config = LongformerConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("O"));
    dummy_label_mapping.insert(1, String::from("LOC"));
    dummy_label_mapping.insert(2, String::from("PER"));
    dummy_label_mapping.insert(3, String::from("ORG"));
    config.id2label = Some(dummy_label_mapping);
    let model = LongformerForTokenClassification::new(&vs.root(), &config);

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
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let model_output = no_grad(|| {
        model.forward_t(
            Some(input_tensor.as_ref()),
            None,
            None,
            None,
            None,
            None,
            false,
        )
    })?;

    assert_eq!(model_output.logits.size(), &[2, 7, 4]);

    Ok(())
}

#[test]
fn longformer_for_question_answering() -> anyhow::Result<()> {
    //    Set-up Question Answering model
    let config = QuestionAnsweringConfig::new(
        ModelType::Longformer,
        RemoteResource::from_pretrained(LongformerModelResources::LONGFORMER_BASE_SQUAD1),
        RemoteResource::from_pretrained(LongformerConfigResources::LONGFORMER_BASE_SQUAD1),
        RemoteResource::from_pretrained(LongformerVocabResources::LONGFORMER_BASE_SQUAD1),
        Some(RemoteResource::from_pretrained(
            LongformerMergesResources::LONGFORMER_BASE_SQUAD1,
        )),
        false,
        None,
        false,
    );

    let qa_model = QuestionAnsweringModel::new(config)?;

    //    Define input
    let question_1 = String::from("Where does Amy live ?");
    let context_1 = String::from("Amy lives in Amsterdam");
    let question_2 = String::from("Where does Eric live");
    let context_2 = String::from("While Amy lives in Amsterdam, Eric is in The Hague.");
    let qa_input_1 = QaInput {
        question: question_1,
        context: context_1,
    };
    let qa_input_2 = QaInput {
        question: question_2,
        context: context_2,
    };

    //    Get answer
    let answers = qa_model.predict(&[qa_input_1, qa_input_2], 1, 32);

    assert_eq!(answers.len(), 2usize);
    assert_eq!(answers[0].len(), 1usize);
    assert_eq!(answers[0][0].start, 12);
    assert_eq!(answers[0][0].end, 22);
    assert!((answers[0][0].score - 0.8060).abs() < 1e-4);
    assert_eq!(answers[0][0].answer, " Amsterdam");
    assert_eq!(answers[1].len(), 1usize);
    assert_eq!(answers[1][0].start, 40);
    assert_eq!(answers[1][0].end, 50);
    assert!((answers[1][0].score - 0.7503).abs() < 1e-4);
    assert_eq!(answers[1][0].answer, " The Hague");

    Ok(())
}
