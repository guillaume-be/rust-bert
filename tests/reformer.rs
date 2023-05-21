use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::reformer::{
    ReformerConfig, ReformerConfigResources, ReformerForQuestionAnswering,
    ReformerForSequenceClassification, ReformerModelResources, ReformerVocabResources,
};
use rust_bert::resources::{LocalResource, RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{MultiThreadedTokenizer, ReformerTokenizer, TruncationStrategy};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::io::Write;
use tch::{nn, no_grad, Device, Tensor};

#[test]
fn test_generation_reformer() -> anyhow::Result<()> {
    // ===================================================
    //    Modify resource to enforce seed
    let config_resource = Box::new(RemoteResource::from_pretrained(
        ReformerConfigResources::CRIME_AND_PUNISHMENT,
    ));

    let original_config_path = config_resource.get_local_path()?;
    let f = File::open(original_config_path).expect("Could not open configuration file.");
    let br = BufReader::new(f);
    let mut config: ReformerConfig =
        serde_json::from_reader(br).expect("could not parse configuration");
    config.hash_seed = Some(42);
    let mut updated_config_file = tempfile::NamedTempFile::new()?;
    let _ = updated_config_file.write_all(serde_json::to_string(&config).unwrap().as_bytes());
    let updated_config_path = updated_config_file.into_temp_path();

    let config_resource = Box::new(LocalResource {
        local_path: updated_config_path.to_path_buf(),
    });
    // ===================================================

    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        ReformerVocabResources::CRIME_AND_PUNISHMENT,
    ));
    let model_resource = Box::new(RemoteResource::from_pretrained(
        ReformerModelResources::CRIME_AND_PUNISHMENT,
    ));
    //    Set-up translation model
    let generation_config = TextGenerationConfig {
        model_type: ModelType::Reformer,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: None,
        min_length: 100,
        max_length: Some(100),
        do_sample: false,
        early_stopping: true,
        no_repeat_ngram_size: 3,
        num_beams: 3,
        num_return_sequences: 1,
        device: Device::Cpu,
        ..Default::default()
    };

    let model = TextGenerationModel::new(generation_config)?;

    let input_context_1 = "The really great men must, I think,";
    let input_context_2 = "It was a gloom winter night, and";
    let output = model.generate(&[input_context_1, input_context_2], None);

    assert_eq!(output.len(), 2);
    assert_eq!(output[0], " The really great men must, I think, anyway waiting for some unknown reason, but Nikodim Fomitch and Ilya Petrovitch looked at him anguish invitable incidently at him. He could not resist an impression which might be setting");
    assert_eq!(output[1], " It was a gloom winter night, and he went out into the street he remembered that he had turned to walked towards the Hay Market. Nastasya was going into a tavern-keeper. He was in the corner; he had come out of the win");

    Ok(())
}

#[test]
fn reformer_for_sequence_classification() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        ReformerConfigResources::CRIME_AND_PUNISHMENT,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        ReformerVocabResources::CRIME_AND_PUNISHMENT,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let tokenizer: ReformerTokenizer =
        ReformerTokenizer::from_file(vocab_path.to_str().unwrap(), true)?;
    let mut config = ReformerConfig::from_file(config_path);
    let mut dummy_label_mapping = HashMap::new();
    dummy_label_mapping.insert(0, String::from("Positive"));
    dummy_label_mapping.insert(1, String::from("Negative"));
    dummy_label_mapping.insert(3, String::from("Neutral"));
    config.id2label = Some(dummy_label_mapping);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let reformer_model = ReformerForSequenceClassification::new(vs.root(), &config)?;

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
        no_grad(|| reformer_model.forward_t(Some(&input_tensor), None, None, None, None, false))?;

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
fn reformer_for_question_answering() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        ReformerConfigResources::CRIME_AND_PUNISHMENT,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        ReformerVocabResources::CRIME_AND_PUNISHMENT,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;

    //    Set-up model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let tokenizer: ReformerTokenizer =
        ReformerTokenizer::from_file(vocab_path.to_str().unwrap(), true)?;
    let mut config = ReformerConfig::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let reformer_model = ReformerForQuestionAnswering::new(vs.root(), &config)?;

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
        no_grad(|| reformer_model.forward_t(Some(&input_tensor), None, None, None, None, false))?;

    assert_eq!(model_output.start_logits.size(), &[2, 19]);
    assert_eq!(model_output.end_logits.size(), &[2, 19]);
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
