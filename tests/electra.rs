use rust_bert::electra::{
    ElectraConfig, ElectraConfigResources, ElectraDiscriminator, ElectraForMaskedLM,
    ElectraModelResources, ElectraVocabResources,
};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{BertTokenizer, MultiThreadedTokenizer, TruncationStrategy};
use rust_tokenizers::vocab::Vocab;
use tch::{nn, no_grad, Device, Tensor};

#[test]
fn electra_masked_lm() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        ElectraConfigResources::BASE_GENERATOR,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        ElectraVocabResources::BASE_GENERATOR,
    ));
    let weights_resource = Box::new(RemoteResource::from_pretrained(
        ElectraModelResources::BASE_GENERATOR,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer =
        BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let mut config = ElectraConfig::from_file(config_path);
    config.output_attentions = Some(true);
    config.output_hidden_states = Some(true);
    let electra_model = ElectraForMaskedLM::new(vs.root(), &config);
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
        no_grad(|| electra_model.forward_t(Some(&input_tensor), None, None, None, None, false));

    //    Decode output
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

    assert_eq!(
        model_output.prediction_scores.size(),
        &[2, 10, config.vocab_size]
    );
    assert_eq!(
        config.num_hidden_layers as usize,
        model_output.all_hidden_states.unwrap().len()
    );
    assert_eq!(
        config.num_hidden_layers as usize,
        model_output.all_attentions.unwrap().len()
    );
    assert_eq!("thing", word_1); // Outputs "person" : "Looks like one [person] is missing"
    assert_eq!("sunny", word_2); // Outputs "pear" : "It was a very nice and [sunny] day"
    Ok(())
}

#[test]
fn electra_discriminator() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        ElectraConfigResources::BASE_DISCRIMINATOR,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        ElectraVocabResources::BASE_DISCRIMINATOR,
    ));
    let weights_resource = Box::new(RemoteResource::from_pretrained(
        ElectraModelResources::BASE_DISCRIMINATOR,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: BertTokenizer =
        BertTokenizer::from_file(vocab_path.to_str().unwrap(), true, true)?;
    let config = ElectraConfig::from_file(config_path);
    let electra_model = ElectraDiscriminator::new(vs.root(), &config);
    vs.load(weights_path)?;

    //    Define input
    let input = ["One Two Three Ten Five Six Seven Eight"];
    let tokenized_input = tokenizer.encode_list(&input, 128, &TruncationStrategy::LongestFirst, 0);
    let max_len = tokenized_input
        .iter()
        .map(|input| input.token_ids.len())
        .max()
        .unwrap();
    let encoded_input = tokenized_input
        .iter()
        .map(|input| input.token_ids.clone())
        .map(|mut input| {
            input.extend(vec![0; max_len - input.len()]);
            input
        })
        .map(|input| Tensor::from_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(encoded_input.as_slice(), 0).to(device);

    //    Forward pass
    let model_output =
        no_grad(|| electra_model.forward_t(Some(&input_tensor), None, None, None, None, false));

    //    Validate model predictions
    let expected_probabilities = vec![
        0.0101, 0.0030, 0.0010, 0.0018, 0.9489, 0.0067, 0.0026, 0.0017, 0.0311, 0.0101,
    ];
    let probabilities = model_output
        .probabilities
        .iter::<f64>()
        .unwrap()
        .collect::<Vec<f64>>();

    assert_eq!(model_output.probabilities.size(), &[10]);
    for (expected, pred) in probabilities.iter().zip(expected_probabilities) {
        assert!((expected - pred).abs() < 1e-4);
    }

    Ok(())
}
