use rust_bert::m2m_100::{
    M2M100Config, M2M100ConfigResources, M2M100MergesResources, M2M100Model, M2M100ModelResources,
    M2M100SourceLanguages, M2M100TargetLanguages, M2M100VocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{M2M100Tokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, Device, Tensor};

#[test]
fn m2m100_lm_model() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = RemoteResource::from_pretrained(M2M100ConfigResources::M2M100_418M);
    let vocab_resource = RemoteResource::from_pretrained(M2M100VocabResources::M2M100_418M);
    let merges_resource = RemoteResource::from_pretrained(M2M100MergesResources::M2M100_418M);
    let weights_resource = RemoteResource::from_pretrained(M2M100ModelResources::M2M100_418M);
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let merges_path = merges_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer = M2M100Tokenizer::from_files(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        false,
    )?;
    let config = M2M100Config::from_file(config_path);
    let m2m100_model = M2M100Model::new(&vs.root() / "model", &config);
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
        m2m100_model.forward_t(Some(&input_tensor), None, None, None, None, None, false);
    assert_eq!(model_output.decoder_output.size(), vec!(1, 5, 1024));
    assert_eq!(
        model_output.encoder_hidden_state.unwrap().size(),
        vec!(1, 5, 1024)
    );
    assert!(
        (model_output.decoder_output.double_value(&[0, 0, 0]) - -2.047429323196411).abs() < 1e-4
    );
    Ok(())
}

#[test]
fn m2m100_translation() -> anyhow::Result<()> {
    let model_resource = RemoteResource::from_pretrained(M2M100ModelResources::M2M100_418M);
    let config_resource = RemoteResource::from_pretrained(M2M100ConfigResources::M2M100_418M);
    let vocab_resource = RemoteResource::from_pretrained(M2M100VocabResources::M2M100_418M);
    let merges_resource = RemoteResource::from_pretrained(M2M100MergesResources::M2M100_418M);

    let source_languages = M2M100SourceLanguages::M2M100_418M;
    let target_languages = M2M100TargetLanguages::M2M100_418M;

    let translation_config = TranslationConfig::new(
        ModelType::M2M100,
        model_resource,
        config_resource,
        vocab_resource,
        Some(merges_resource),
        source_languages,
        target_languages,
        Device::cuda_if_available(),
    );
    let model = TranslationModel::new(translation_config)?;

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
    assert_eq!(outputs[1], " Esta frase se traducirá en varios idiomas.");
    assert_eq!(outputs[2], " यह वाक्यांश कई भाषाओं में अनुवादित किया जाएगा।");

    Ok(())
}
