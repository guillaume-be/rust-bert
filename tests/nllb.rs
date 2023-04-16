use rust_bert::nllb::{
    NLLBConfigResources, NLLBLanguages, NLLBMergeResources, NLLBResources, NLLBVocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
use rust_bert::resources::RemoteResource;
use tch::Device;

#[test]
fn nllb_translation() -> anyhow::Result<()> {
    let model_resource = RemoteResource::from_pretrained(NLLBResources::NLLB_600M_DISTILLED);
    let config_resource = RemoteResource::from_pretrained(NLLBConfigResources::NLLB_600M_DISTILLED);
    let vocab_resource = RemoteResource::from_pretrained(NLLBVocabResources::NLLB_600M_DISTILLED);
    let merges_resource = RemoteResource::from_pretrained(NLLBMergeResources::NLLB_600M_DISTILLED);
    // let special_map = RemoteResource::from_pretrained(NLLBSpecialMap::NLLB_600M_DISTILLED);

    let source_languages = NLLBLanguages::NLLB;
    let target_languages = NLLBLanguages::NLLB;

    let translation_config = TranslationConfig::new(
        ModelType::NLLB,
        model_resource,
        config_resource,
        vocab_resource,
        Some(merges_resource),
        source_languages,
        target_languages,
        Device::Cpu,
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
    assert_eq!(outputs[1], " Esta frase será traducida a varios idiomas.");
    assert_eq!(outputs[2], " यह वाक्य कई भाषाओं में अनुवादित किया जाएगा।");

    Ok(())
}
