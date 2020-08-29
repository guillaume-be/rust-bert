use rust_bert::bart::{
    BartConfig, BartConfigResources, BartMergesResources, BartModel, BartModelResources,
    BartVocabResources,
};
use rust_bert::pipelines::summarization::{SummarizationConfig, SummarizationModel};
use rust_bert::resources::{download_resource, RemoteResource, Resource};
use rust_bert::Config;
use rust_tokenizers::{RobertaTokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, Device, Tensor};

#[test]
#[cfg_attr(not(feature = "all-tests"), ignore)]
fn bart_lm_model() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource =
        Resource::Remote(RemoteResource::from_pretrained(BartConfigResources::BART));
    let vocab_resource =
        Resource::Remote(RemoteResource::from_pretrained(BartVocabResources::BART));
    let merges_resource =
        Resource::Remote(RemoteResource::from_pretrained(BartMergesResources::BART));
    let weights_resource =
        Resource::Remote(RemoteResource::from_pretrained(BartModelResources::BART));
    let config_path = download_resource(&config_resource)?;
    let vocab_path = download_resource(&vocab_resource)?;
    let merges_path = download_resource(&merges_resource)?;
    let weights_path = download_resource(&weights_resource)?;

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
    let bart_model = BartModel::new(&vs.root(), &config, false);
    vs.load(weights_path)?;

    //    Define input
    let input = ["One two three four"];
    let tokenized_input =
        tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
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
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let (output, encoder_outputs, _, _, _, _, _) =
        bart_model.forward_t(Some(&input_tensor), None, None, None, None, None, false);

    assert_eq!(output.size(), vec!(1, 6, 1024));
    assert_eq!(encoder_outputs.size(), vec!(1, 6, 1024));
    assert!((output.double_value(&[0, output.size()[1] - 1, 0]) - (-0.2420)).abs() < 1e-4);
    Ok(())
}

#[test]
#[cfg_attr(not(feature = "all-tests"), ignore)]
fn bart_summarization_greedy() -> anyhow::Result<()> {
    //    Set-up masked LM model
    let summarization_config = SummarizationConfig {
        num_beams: 1,
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
    assert_eq!(output[0], "Scientists have found water vapour on K2-18b, a planet 110 light-years from Earth. This \
    is the first such discovery in a planet in its star's habitable zone. The planet is not too hot and not too cold \
    for liquid water to exist.");

    Ok(())
}

#[test]
#[cfg_attr(not(feature = "all-tests"), ignore)]
fn bart_summarization_beam_search() -> anyhow::Result<()> {
    //    Set-up masked LM model
    let summarization_config = SummarizationConfig {
        num_beams: 3,
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
    assert_eq!(output[0], "K2-18b, a planet in its star's habitable zone, has water vapour in its atmosphere. \
    This is the first such discovery in a planet not too hot and not too cold for liquid water to exist. The \
    Montreal team used data from the NASA's Hubble telescope to assess changes in the light coming from the \
    star as the planet passed between it and Earth.");

    Ok(())
}
