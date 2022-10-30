use rust_bert::marian::{
    MarianConfigResources, MarianModelResources, MarianSourceLanguages, MarianSpmResources,
    MarianTargetLanguages, MarianVocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::translation::{
    Language, TranslationConfig, TranslationModel, TranslationModelBuilder,
};
use rust_bert::resources::RemoteResource;
use tch::Device;

#[test]
// #[cfg_attr(not(feature = "all-tests"), ignore)]
fn test_translation() -> anyhow::Result<()> {
    //    Set-up translation model
    let model_resource = RemoteResource::from_pretrained(MarianModelResources::ENGLISH2ROMANCE);
    let config_resource = RemoteResource::from_pretrained(MarianConfigResources::ENGLISH2ROMANCE);
    let vocab_resource = RemoteResource::from_pretrained(MarianVocabResources::ENGLISH2ROMANCE);
    let merges_resource = RemoteResource::from_pretrained(MarianSpmResources::ENGLISH2ROMANCE);

    let source_languages = MarianSourceLanguages::ENGLISH2ROMANCE;
    let target_languages = MarianTargetLanguages::ENGLISH2ROMANCE;

    let translation_config = TranslationConfig::new(
        ModelType::Marian,
        model_resource,
        config_resource,
        vocab_resource,
        Some(merges_resource),
        source_languages,
        target_languages,
        Device::cuda_if_available(),
    );
    let model = TranslationModel::new(translation_config)?;

    let input_context_1 = "The quick brown fox jumps over the lazy dog";
    let input_context_2 = "The dog did not wake up";

    let outputs = model.translate(&[input_context_1, input_context_2], None, Language::French)?;

    assert_eq!(outputs.len(), 2);
    assert_eq!(
        outputs[0],
        " Le rapide renard brun saute sur le chien paresseux"
    );
    assert_eq!(outputs[1], " Le chien ne s'est pas réveillé");

    Ok(())
}

#[test]
// #[cfg_attr(not(feature = "all-tests"), ignore)]
fn test_translation_builder() -> anyhow::Result<()> {
    let model = TranslationModelBuilder::new()
        .with_device(Device::cuda_if_available())
        .with_model_type(ModelType::Marian)
        .with_source_languages(vec![Language::English])
        .with_target_languages(vec![Language::French])
        .create_model()?;

    let input_context_1 = "The quick brown fox jumps over the lazy dog";
    let input_context_2 = "The dog did not wake up";

    let outputs = model.translate(&[input_context_1, input_context_2], None, Language::French)?;

    assert_eq!(outputs.len(), 2);
    assert_eq!(
        outputs[0],
        " Le rapide renard brun saute sur le chien paresseux"
    );
    assert_eq!(outputs[1], " Le chien ne s'est pas réveillé");

    Ok(())
}
