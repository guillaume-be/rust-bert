#[macro_use]
extern crate criterion;

use criterion::{black_box, Criterion};
use rust_bert::gpt2::{
    Gpt2ConfigResources, Gpt2MergesResources, Gpt2ModelResources, Gpt2VocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
use rust_bert::resources::RemoteResource;
use std::time::{Duration, Instant};
use tch::Device;

fn create_text_generation_model() -> TextGenerationModel {
    let config = TextGenerationConfig {
        model_type: ModelType::GPT2,
        model_resource: Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2)),
        config_resource: Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2)),
        vocab_resource: Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2)),
        merges_resource: Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2)),
        min_length: 0,
        max_length: 30,
        do_sample: true,
        early_stopping: false,
        num_beams: 5,
        temperature: 1.0,
        top_k: 0,
        top_p: 0.9,
        repetition_penalty: 1.0,
        length_penalty: 1.0,
        no_repeat_ngram_size: 3,
        num_beam_groups: None,
        diversity_penalty: None,
        num_return_sequences: 5,
        device: Device::cuda_if_available(),
    };
    TextGenerationModel::new(config).unwrap()
}

fn generation_forward_pass(iters: u64, model: &TextGenerationModel, data: &[&str]) -> Duration {
    let mut duration = Duration::new(0, 0);
    for _i in 0..iters {
        let start = Instant::now();
        let _ = model.generate(data, None);
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

fn bench_generation(c: &mut Criterion) {
    //    Set-up summarization model
    unsafe {
        torch_sys::dummy_cuda_dependency();
    }
    let model = create_text_generation_model();

    //    Define input
    let input = ["Hello, I'm a language model,"];
    c.bench_function("Generation", |b| {
        b.iter_custom(|iters| black_box(generation_forward_pass(iters, &model, &input)))
    });
}

criterion_group! {
name = benches;
config = Criterion::default().sample_size(10);
targets = bench_generation
}

criterion_main!(benches);
