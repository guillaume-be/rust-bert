#[macro_use]
extern crate criterion;

use criterion::{black_box, Criterion};
// use rust_bert::pipelines::common::ModelType;
// use rust_bert::pipelines::translation::TranslationOption::{Marian, T5};
use rust_bert::pipelines::translation::{OldLanguage, TranslationConfig, TranslationModel};
// use rust_bert::resources::{LocalResource, Resource};
use std::time::{Duration, Instant};
use tch::Device;

fn create_translation_model() -> TranslationModel {
    let config =
        TranslationConfig::new(OldLanguage::EnglishToFrenchV2, Device::cuda_if_available());
    // let config = TranslationConfig::new_from_resources(
    //     Resource::Local(LocalResource {
    //         local_path: "E:/Coding/cache/rustbert/marian-mt-en-es/model.ot".into(),
    //     }),
    //     Resource::Local(LocalResource {
    //         local_path: "E:/Coding/cache/rustbert/marian-mt-en-es/config.json".into(),
    //     }),
    //     Resource::Local(LocalResource {
    //         local_path: "E:/Coding/cache/rustbert/marian-mt-en-es/vocab.json".into(),
    //     }),
    //     Resource::Local(LocalResource {
    //         local_path: "E:/Coding/cache/rustbert/marian-mt-en-es/spiece.model".into(),
    //     }),
    //     None,
    //     Device::cuda_if_available(),
    //     ModelType::Marian,
    // );
    TranslationModel::new(config).unwrap()
}

fn translation_forward_pass(iters: u64, model: &TranslationModel, data: &[&str]) -> Duration {
    let mut duration = Duration::new(0, 0);
    for _i in 0..iters {
        let start = Instant::now();
        let _ = model.translate(data);
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

fn translation_load_model(iters: u64) -> Duration {
    let mut duration = Duration::new(0, 0);
    for _i in 0..iters {
        let start = Instant::now();
        let config =
            TranslationConfig::new(OldLanguage::EnglishToFrenchV2, Device::cuda_if_available());
        // let config = TranslationConfig::new_from_resources(
        //     Resource::Local(LocalResource {
        //         local_path: "E:/Coding/cache/rustbert/marian-mt-en-es/model.ot".into(),
        //     }),
        //     Resource::Local(LocalResource {
        //         local_path: "E:/Coding/cache/rustbert/marian-mt-en-es/config.json".into(),
        //     }),
        //     Resource::Local(LocalResource {
        //         local_path: "E:/Coding/cache/rustbert/marian-mt-en-es/vocab.json".into(),
        //     }),
        //     Resource::Local(LocalResource {
        //         local_path: "E:/Coding/cache/rustbert/marian-mt-en-es/spiece.model".into(),
        //     }),
        //     None,
        //     Device::cuda_if_available(),
        //     ModelType::Marian,
        // );
        TranslationModel::new(config).unwrap();
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

fn bench_squad(c: &mut Criterion) {
    //    Set-up summarization model
    unsafe {
        torch_sys::dummy_cuda_dependency();
    }
    let model = create_translation_model();

    //    Define input
    let input = [
        "In findings published Tuesday in Cornell University's arXiv by a team of scientists from the University of Montreal and a separate report published Wednesday in Nature Astronomy by a team from University College London (UCL), the presence of water vapour was confirmed in the atmosphere of K2-18b, a planet circling a star in the constellation Leo.",
        "This is the first such discovery in a planet in its star's habitable zone — not too hot and not too cold for liquid water to exist. The Montreal team, led by Björn Benneke, used data from the NASA\'s Hubble telescope to assess changes in the light coming from K2-18b's star as the planet passed between it and Earth.",
        "They found that certain wavelengths of light, which are usually absorbed by water, weakened when the planet was in the way, indicating not only does K2-18b have an atmosphere, but the atmosphere contains water in vapour form.",
        "The team from UCL then analyzed the Montreal team's data using their own software and confirmed their conclusion.",
        "This was not the first time scientists have found signs of water on an exoplanet, but previous discoveries were made on planets with high temperatures or other pronounced differences from Earth.",
        "This is the first potentially habitable planet where the temperature is right and where we now know there is water,\" said UCL astronomer Angelos Tsiaras.",
        "It's the best candidate for habitability right now.\" \"It's a good sign\", said Ryan Cloutier of the Harvard–Smithsonian Center for Astrophysics, who was not one of either study's authors.",
        "Overall,\" he continued, \"the presence of water in its atmosphere certainly improves the prospect of K2-18b being a potentially habitable planet, but further observations will be required to say for sure. \"",
        "K2-18b was first identified in 2015 by the Kepler space telescope.",
        "It is about 110 light-years from Earth and larger but less dense.",
    ];
    // (New sample credits: [WikiNews](https://en.wikinews.org/wiki/Astronomers_find_water_vapour_in_atmosphere_of_exoplanet_K2-18b))
    c.bench_function("Translation forward pass", |b| {
        b.iter_custom(|iters| black_box(translation_forward_pass(iters, &model, &input)))
    });

    c.bench_function("Load model", |b| {
        b.iter_custom(|iters| black_box(translation_load_model(iters)))
    });
}

criterion_group! {
name = benches;
config = Criterion::default().sample_size(10);
targets = bench_squad
}

criterion_main!(benches);
