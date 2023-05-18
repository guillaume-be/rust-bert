use rust_bert::bart::{
    BartConfig, BartConfigResources, BartMergesResources, BartModel, BartModelResources,
    BartVocabResources,
};
use rust_bert::pipelines::summarization::{SummarizationConfig, SummarizationModel};
use rust_bert::pipelines::zero_shot_classification::{
    ZeroShotClassificationConfig, ZeroShotClassificationModel,
};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::{Config, RustBertError};
use rust_tokenizers::tokenizer::{RobertaTokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, Device, Tensor};

#[test]
fn bart_lm_model() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Box::new(RemoteResource::from_pretrained(
        BartConfigResources::DISTILBART_CNN_6_6,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        BartVocabResources::DISTILBART_CNN_6_6,
    ));
    let merges_resource = Box::new(RemoteResource::from_pretrained(
        BartMergesResources::DISTILBART_CNN_6_6,
    ));
    let weights_resource = Box::new(RemoteResource::from_pretrained(
        BartModelResources::DISTILBART_CNN_6_6,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let merges_path = merges_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: RobertaTokenizer = RobertaTokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        false,
        false,
    )?;
    let config = BartConfig::from_file(config_path);
    let bart_model = BartModel::new(&vs.root() / "model", &config);
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
        bart_model.forward_t(Some(&input_tensor), None, None, None, None, None, false);
    assert_eq!(model_output.decoder_output.size(), vec!(1, 6, 1024));
    assert_eq!(
        model_output.encoder_hidden_state.unwrap().size(),
        vec!(1, 6, 1024)
    );
    assert!((model_output.decoder_output.double_value(&[0, 0, 0]) - 0.2610).abs() < 1e-4);
    Ok(())
}

#[test]
fn bart_summarization_greedy() -> anyhow::Result<()> {
    let config_resource = Box::new(RemoteResource::from_pretrained(
        BartConfigResources::DISTILBART_CNN_6_6,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        BartVocabResources::DISTILBART_CNN_6_6,
    ));
    let merges_resource = Box::new(RemoteResource::from_pretrained(
        BartMergesResources::DISTILBART_CNN_6_6,
    ));
    let model_resource = Box::new(RemoteResource::from_pretrained(
        BartModelResources::DISTILBART_CNN_6_6,
    ));
    let summarization_config = SummarizationConfig {
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        num_beams: 1,
        length_penalty: 1.0,
        min_length: 56,
        max_length: Some(142),
        device: Device::Cpu,
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

    //    Credits: WikiNews, CC BY 2.5 license (https://en.wikinews.org/wiki/Astronomers_find_water_vapour_in_atmosphere_of_exoplanet_K2-18b)
    let output = model.summarize(&input);

    assert_eq!(output.len(), 1);
    assert_eq!(output[0], " K2-18b is not too hot and not too cold for liquid water to exist. \
    This is the first such discovery in a planet in its star's habitable zone. \
    The presence of water vapour was confirmed in the atmosphere of K2, a planet circling a star in the constellation Leo.");

    Ok(())
}

#[test]
fn bart_summarization_beam_search() -> anyhow::Result<()> {
    let config_resource = Box::new(RemoteResource::from_pretrained(
        BartConfigResources::DISTILBART_CNN_6_6,
    ));
    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        BartVocabResources::DISTILBART_CNN_6_6,
    ));
    let merges_resource = Box::new(RemoteResource::from_pretrained(
        BartMergesResources::DISTILBART_CNN_6_6,
    ));
    let model_resource = Box::new(RemoteResource::from_pretrained(
        BartModelResources::DISTILBART_CNN_6_6,
    ));
    let summarization_config = SummarizationConfig {
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        num_beams: 4,
        min_length: 56,
        max_length: Some(142),
        length_penalty: 1.0,
        device: Device::Cpu,
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
a potentially habitable planet, but further observations will be required to say for sure. \"
K2-18b was first identified in 2015 by the Kepler space telescope. It is about 110 light-years from Earth and larger \
but less dense. Its star, a red dwarf, is cooler than the Sun, but the planet's orbit is much closer, such that a year \
on K2-18b lasts 33 Earth days. According to The Guardian, astronomers were optimistic that NASA's James Webb space \
telescope — scheduled for launch in 2021 — and the European Space Agency's 2028 ARIEL program, could reveal more \
about exoplanets like K2-18b."];

    //    Credits: WikiNews, CC BY 2.5 license (https://en.wikinews.org/wiki/Astronomers_find_water_vapour_in_atmosphere_of_exoplanet_K2-18b)
    let output = model.summarize(&input);

    assert_eq!(output.len(), 1);
    assert_eq!(output[0], " K2-18b, a planet circling a star in the constellation Leo, is not too hot and not too cold for liquid water to exist. \
    This is the first such discovery in a planet in its star's habitable zone. \
    It is not the first time scientists have found signs of water on an exoplanet.");

    Ok(())
}

#[test]
#[cfg_attr(not(feature = "all-tests"), ignore)]
fn bart_zero_shot_classification() -> anyhow::Result<()> {
    //    Set-up model
    let zero_shot_config = ZeroShotClassificationConfig {
        device: Device::Cpu,
        ..Default::default()
    };
    let sequence_classification_model = ZeroShotClassificationModel::new(zero_shot_config)?;

    let input_sentence = "Who are you voting for in 2020?";
    let input_sequence_2 = "The prime minister has announced a stimulus package which was widely criticized by the opposition.";
    let candidate_labels = &["politics", "public health", "economy", "sports"];

    let output = sequence_classification_model.predict(
        [input_sentence, input_sequence_2],
        candidate_labels,
        Some(Box::new(|label: &str| {
            format!("This example is about {label}.")
        })),
        128,
    )?;

    assert_eq!(output.len(), 2);

    // Prediction scores
    assert_eq!(output[0].text, "politics");
    assert!((output[0].score - 0.9630).abs() < 1e-4);
    assert_eq!(output[1].text, "economy");
    assert!((output[1].score - 0.6416).abs() < 1e-4);
    Ok(())
}

#[test]
#[cfg_attr(not(feature = "all-tests"), ignore)]
fn bart_zero_shot_classification_try_error() -> anyhow::Result<()> {
    //    Set-up model
    let zero_shot_config = ZeroShotClassificationConfig {
        device: Device::Cpu,
        ..Default::default()
    };
    let sequence_classification_model = ZeroShotClassificationModel::new(zero_shot_config)?;

    let output = sequence_classification_model.predict(
        [],
        [],
        Some(Box::new(|label: &str| {
            format!("This example is about {label}.")
        })),
        128,
    );

    let output_is_error = match output {
        Err(RustBertError::ValueError(_)) => true,
        _ => unreachable!(),
    };
    assert!(output_is_error);

    Ok(())
}

#[test]
#[cfg_attr(not(feature = "all-tests"), ignore)]
fn bart_zero_shot_classification_multilabel() -> anyhow::Result<()> {
    // Set-up model
    let zero_shot_config = ZeroShotClassificationConfig {
        device: Device::Cpu,
        ..Default::default()
    };
    let sequence_classification_model = ZeroShotClassificationModel::new(zero_shot_config)?;

    let input_sentence = "Who are you voting for in 2020?";
    let input_sequence_2 = "The prime minister has announced a stimulus package which was widely criticized by the opposition.";
    let candidate_labels = &["politics", "public health", "economy", "sports"];

    let output = sequence_classification_model.predict_multilabel(
        [input_sentence, input_sequence_2],
        candidate_labels,
        Some(Box::new(|label: &str| {
            format!("This example is about {label}.")
        })),
        128,
    )?;

    assert_eq!(output.len(), 2);
    assert_eq!(output[0].len(), candidate_labels.len());
    // First sentence label scores
    assert_eq!(output[0][0].text, "politics");
    assert!((output[0][0].score - 0.9805).abs() < 1e-4);
    assert_eq!(output[0][1].text, "public health");
    assert!((output[0][1].score - 0.0130).abs() < 1e-4);
    assert_eq!(output[0][2].text, "economy");
    assert!((output[0][2].score - 0.0255).abs() < 1e-4);
    assert_eq!(output[0][3].text, "sports");
    assert!((output[0][3].score - 0.0013).abs() < 1e-4);

    // Second sentence label scores
    assert_eq!(output[1][0].text, "politics");
    assert!((output[1][0].score - 0.9432).abs() < 1e-4);
    assert_eq!(output[1][1].text, "public health");
    assert!((output[1][1].score - 0.0045).abs() < 1e-4);
    assert_eq!(output[1][2].text, "economy");
    assert!((output[1][2].score - 0.9851).abs() < 1e-4);
    assert_eq!(output[1][3].text, "sports");
    assert!((output[1][3].score - 0.0004).abs() < 1e-4);
    Ok(())
}

#[test]
#[cfg_attr(not(feature = "all-tests"), ignore)]
fn bart_zero_shot_classification_multilabel_try_error() -> anyhow::Result<()> {
    //    Set-up model
    let zero_shot_config = ZeroShotClassificationConfig {
        device: Device::Cpu,
        ..Default::default()
    };
    let sequence_classification_model = ZeroShotClassificationModel::new(zero_shot_config)?;

    let output = sequence_classification_model.predict_multilabel(
        [],
        [],
        Some(Box::new(|label: &str| {
            format!("This example is about {label}.")
        })),
        128,
    );

    let output_is_error = match output {
        Err(RustBertError::ValueError(_)) => true,
        _ => unreachable!(),
    };
    assert!(output_is_error);

    Ok(())
}
