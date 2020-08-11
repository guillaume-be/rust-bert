use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::translation::{TranslationConfig, TranslationModel};
use rust_bert::resources::{RemoteResource, Resource};
use rust_bert::t5::{T5ConfigResources, T5ModelResources, T5VocabResources};
use tch::Device;

#[test]
fn test_translation_t5() -> anyhow::Result<()> {
    //    Set-up translation model
    let translation_config = TranslationConfig::new_from_resources(
        Resource::Remote(RemoteResource::from_pretrained(T5ModelResources::T5_SMALL)),
        Resource::Remote(RemoteResource::from_pretrained(T5ConfigResources::T5_SMALL)),
        Resource::Remote(RemoteResource::from_pretrained(T5VocabResources::T5_SMALL)),
        Resource::Remote(RemoteResource::from_pretrained(T5VocabResources::T5_SMALL)),
        Some("translate English to French: ".to_string()),
        Device::cuda_if_available(),
        ModelType::T5,
    );
    let model = TranslationModel::new(translation_config)?;

    let input_context = "The quick brown fox jumps over the lazy dog";

    let output = model.translate(&[input_context]);

    assert_eq!(
        output[0],
        " Le renard brun rapide saute au-dessus du chien paresseux."
    );

    Ok(())
}
