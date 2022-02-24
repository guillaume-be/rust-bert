#[macro_use]
extern crate criterion;

use criterion::{black_box, Criterion};
use rust_bert::bert::{BertConfigResources, BertModelResources, BertVocabResources};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::question_answering::{
    squad_processor, QaInput, QuestionAnsweringConfig, QuestionAnsweringModel,
};
use rust_bert::resources::RemoteResource;
use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

static BATCH_SIZE: usize = 64;

fn create_qa_model() -> QuestionAnsweringModel {
    let config = QuestionAnsweringConfig::new(
        ModelType::Bert,
        RemoteResource::from_pretrained(BertModelResources::BERT_QA),
        RemoteResource::from_pretrained(BertConfigResources::BERT_QA),
        RemoteResource::from_pretrained(BertVocabResources::BERT_QA),
        None,  //merges resource only relevant with ModelType::Roberta
        false, //lowercase
        false,
        None,
    );
    QuestionAnsweringModel::new(config).unwrap()
}

fn squad_forward_pass(
    iters: u64,
    model: &QuestionAnsweringModel,
    squad_data: &[QaInput],
) -> Duration {
    let mut duration = Duration::new(0, 0);
    let batch_size = BATCH_SIZE;
    let mut output = vec![];
    for _i in 0..iters {
        let start = Instant::now();
        for batch in squad_data.chunks(batch_size) {
            output.push(model.predict(batch, 1, 64));
        }
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

fn qa_load_model(iters: u64) -> Duration {
    let mut duration = Duration::new(0, 0);
    for _i in 0..iters {
        let start = Instant::now();
        let config = QuestionAnsweringConfig::new(
            ModelType::Bert,
            RemoteResource::from_pretrained(BertModelResources::BERT_QA),
            RemoteResource::from_pretrained(BertConfigResources::BERT_QA),
            RemoteResource::from_pretrained(BertVocabResources::BERT_QA),
            None,  //merges resource only relevant with ModelType::Roberta
            false, //lowercase
            false,
            None,
        );
        let _ = QuestionAnsweringModel::new(config).unwrap();
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

fn bench_squad(c: &mut Criterion) {
    //    Set-up QA model
    let model = create_qa_model();
    unsafe {
        torch_sys::dummy_cuda_dependency();
    }
    //    Define input
    let mut squad_path = PathBuf::from(env::var("squad_dataset")
        .expect("Please set the \"squad_dataset\" environment variable pointing to the SQuAD dataset folder"));
    squad_path.push("dev-v2.0.json");
    let mut qa_inputs = squad_processor(squad_path);
    qa_inputs.truncate(1000);

    c.bench_function("SQuAD forward pass", |b| {
        b.iter_custom(|iters| black_box(squad_forward_pass(iters, &model, &qa_inputs)))
    });

    c.bench_function("Load model", |b| {
        b.iter_custom(|iters| black_box(qa_load_model(iters)))
    });
}

criterion_group! {
name = benches;
config = Criterion::default().sample_size(10);
targets = bench_squad
}

criterion_main!(benches);
