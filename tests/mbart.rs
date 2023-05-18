use rust_bert::mbart::{
    MBartConfig, MBartConfigResources, MBartModel, MBartModelResources, MBartVocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::translation::{Language, TranslationModelBuilder};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{MBart50Tokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, Device, Tensor};

#[test]
fn mbart_lm_model() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        MBartConfigResources::MBART50_MANY_TO_MANY,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        MBartVocabResources::MBART50_MANY_TO_MANY,
    ));
    let weights_resource = Box::new(RemoteResource::from_pretrained(
        MBartModelResources::MBART50_MANY_TO_MANY,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer = MBart50Tokenizer::from_file(vocab_path.to_str().unwrap(), false)?;
    let config = MBartConfig::from_file(config_path);
    let mbart_model = MBartModel::new(&vs.root() / "model", &config);
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
        .map(|input| Tensor::from_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let model_output =
        mbart_model.forward_t(Some(&input_tensor), None, None, None, None, None, false);
    assert_eq!(model_output.decoder_output.size(), vec!(1, 5, 1024));
    assert_eq!(
        model_output.encoder_hidden_state.unwrap().size(),
        vec!(1, 5, 1024)
    );
    assert!((model_output.decoder_output.double_value(&[0, 0, 0]) - -0.8936).abs() < 1e-4);
    Ok(())
}

#[test]
fn mbart_translation() -> anyhow::Result<()> {
    let model = TranslationModelBuilder::new()
        .with_device(Device::cuda_if_available())
        .with_model_type(ModelType::MBart)
        .create_model()?;

    let source_sentence = "This sentence will be translated in multiple languages.";

    let mut outputs = Vec::new();
    outputs.extend(model.translate(&[source_sentence], Language::English, Language::French)?);
    outputs.extend(model.translate(&[source_sentence], Language::English, Language::Spanish)?);
    outputs.extend(model.translate(&[source_sentence], Language::English, Language::Hindi)?);

    assert_eq!(outputs.len(), 3);
    assert_eq!(
        outputs[0],
        " Cette phrase sera traduite en plusieurs langues."
    );
    assert_eq!(
        outputs[1],
        " Esta frase será traducida en múltiples idiomas."
    );
    assert_eq!(outputs[2], " यह वाक्य कई भाषाओं में अनुवाद किया जाएगा.");

    Ok(())
}
