use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::summarization::{SummarizationConfig, SummarizationModel};
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
        Some("translate English to French:".to_string()),
        Device::cuda_if_available(),
        ModelType::T5,
    );
    let model = TranslationModel::new(translation_config)?;

    let input_context = "The quick brown fox jumps over the lazy dog.";

    let output = model.translate(&[input_context]);

    assert_eq!(
        output[0],
        " Le renard brun rapide saute au-dessus du chien paresseux."
    );

    Ok(())
}

#[test]
fn test_summarization_t5() -> anyhow::Result<()> {
    //    Set-up translation model
    let summarization_config = SummarizationConfig {
        model_type: ModelType::T5,
        model_resource: Resource::Remote(RemoteResource::from_pretrained(
            T5ModelResources::T5_SMALL,
        )),
        config_resource: Resource::Remote(RemoteResource::from_pretrained(
            T5ConfigResources::T5_SMALL,
        )),
        vocab_resource: Resource::Remote(RemoteResource::from_pretrained(
            T5VocabResources::T5_SMALL,
        )),
        merges_resource: Resource::Remote(RemoteResource::from_pretrained(
            T5VocabResources::T5_SMALL,
        )),
        min_length: 30,
        max_length: 200,
        early_stopping: true,
        num_beams: 4,
        length_penalty: 2.0,
        ..Default::default()
    };
    let model = SummarizationModel::new(summarization_config)?;

    let input = ["In findings published Tuesday in Cornell University's arXiv by a team of scientists \
from the University of Montreal and a separate report published Wednesday in Nature Astronomy by a team \
from University College London (UCL), the presence of water vapour was confirmed in the atmosphere of K2-18b, \
a planet circling a star in the constellation Leo. This is the first such discovery in a planet in its star's \
habitable zone — not too hot and not too cold for liquid water to exist. The Montreal team, led by Björn Benneke, \
used data from the NASA's Hubble telescope to assess changes in the light coming from K2-18b's star as the planet \
passed between it and Earth. They found that certain wavelengths of light, which are usually absorbed by water, \
weakened when the planet was in the way, indicating not only does K2-18b have an atmosphere, but the atmosphere \
contains water in vapour form. The team from UCL then analyzed the Montreal team's data using their own software \
and confirmed their conclusion. This was not the first time scientists have found signs of water on an exoplanet, \
but previous discoveries were made on planets with high temperatures or other pronounced differences from Earth. \
\"This is the first potentially habitable planet where the temperature is right and where we now know there is water,\" \
said UCL astronomer Angelos Tsiaras. \"It's the best candidate for habitability right now.\" \"It's a good sign\", \
said Ryan Cloutier of the Harvard–Smithsonian Center for Astrophysics, who was not one of either study's authors. \
\"Overall,\" he continued, \"the presence of water in its atmosphere certainly improves the prospect of K2-18b being \
a potentially habitable planet, but further observations will be required to say for sure. \" \
K2-18b was first identified in 2015 by the Kepler space telescope. It is about 110 light-years from Earth and larger \
but less dense. Its star, a red dwarf, is cooler than the Sun, but the planet's orbit is much closer, such that a year \
on K2-18b lasts 33 Earth days. According to The Guardian, astronomers were optimistic that NASA's James Webb space \
telescope — scheduled for launch in 2021 — and the European Space Agency's 2028 ARIEL program, could reveal more \
about exoplanets like K2-18b."];

    let output = model.summarize(&input);

    assert_eq! (
    output[0],
    " the presence of water vapour was confirmed in the atmosphere of K2-18b. \
    this is the first such discovery in a planet in its star's habitable zone. \
    previous discoveries were made on planets with high temperatures or other pronounced differences from Earth."
    );

    Ok(())
}
