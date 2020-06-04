use rust_bert::pipelines::translation::{TranslationConfig, Language, TranslationModel};
use tch::Device;

#[test]
#[cfg_attr(not(feature = "all-tests"), ignore)]
fn test_translation() -> failure::Fallible<()> {

//    Set-up translation model
    let translation_config =  TranslationConfig::new(Language::EnglishToFrench, Device::Cpu);
    let mut model = TranslationModel::new(translation_config)?;

    let input_context_1 = "The quick brown fox jumps over the lazy dog";
    let input_context_2 = "The dog did not wake up";

    let output = model.translate(&[input_context_1, input_context_2]);

    assert_eq!(output.len(), 2);
    assert_eq!(output[0], " Le rapide renard brun saute sur le chien paresseux");
    assert_eq!(output[1], " Le chien ne s'est pas réveillé.");

    Ok(())
}