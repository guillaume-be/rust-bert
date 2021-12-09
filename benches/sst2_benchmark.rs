#[macro_use]
extern crate criterion;

use criterion::Criterion;
use rust_bert::pipelines::sentiment::SentimentModel;
use rust_bert::pipelines::sequence_classification::SequenceClassificationConfig;
use serde::Deserialize;
use std::error::Error;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::{env, fs};
use tch::Device;

static BATCH_SIZE: usize = 64;

fn create_sentiment_model() -> SentimentModel {
    let config = SequenceClassificationConfig {
        device: Device::cuda_if_available(),
        ..Default::default()
    };
    SentimentModel::new(config).unwrap()
}

fn sst2_forward_pass(iters: u64, model: &SentimentModel, sst2_data: &[String]) -> Duration {
    let mut duration = Duration::new(0, 0);
    let batch_size = BATCH_SIZE;
    let mut output = vec![];
    for _i in 0..iters {
        let start = Instant::now();
        for batch in sst2_data.chunks(batch_size) {
            output.push(
                model.predict(
                    batch
                        .iter()
                        .map(|v| v.as_str())
                        .collect::<Vec<&str>>()
                        .as_slice(),
                ),
            );
        }
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

#[derive(Debug, Deserialize)]
struct Record {
    sentence: String,
}

fn ss2_processor(file_path: PathBuf) -> Result<Vec<String>, Box<dyn Error>> {
    let file = fs::File::open(file_path).expect("unable to open file");
    let mut csv = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b'\t')
        .from_reader(file);
    let mut records = Vec::new();
    for result in csv.deserialize() {
        let record: Record = result?;
        records.push(record.sentence);
    }
    Ok(records)
}

fn sst2_load_model(iters: u64) -> Duration {
    let mut duration = Duration::new(0, 0);
    for _i in 0..iters {
        let start = Instant::now();
        let config = SequenceClassificationConfig {
            device: Device::cuda_if_available(),
            ..Default::default()
        };
        let _ = SentimentModel::new(config).unwrap();
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

fn bench_sst2(c: &mut Criterion) {
    //    Set-up classifier
    let model = create_sentiment_model();
    unsafe {
        torch_sys::dummy_cuda_dependency();
    }
    //    Define input
    let mut sst2_path = PathBuf::from(env::var("SST2_PATH")
        .expect("Please set the \"squad_dataset\" environment variable pointing to the SQuAD dataset folder"));
    sst2_path.push("train.tsv");
    let mut inputs = ss2_processor(sst2_path).unwrap();
    inputs.truncate(2000);

    c.bench_function("SST2 forward pass", |b| {
        b.iter_custom(|iters| sst2_forward_pass(iters, &model, &inputs))
    });

    c.bench_function("Load model", |b| b.iter_custom(sst2_load_model));
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_sst2
}

criterion_main!(benches);
