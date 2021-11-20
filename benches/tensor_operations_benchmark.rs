#[macro_use]
extern crate criterion;

use criterion::{black_box, Criterion};
use std::time::{Duration, Instant};
use tch::kind::Kind;
use tch::{Device, Tensor};

fn matrix_multiply(iters: u64, input: &Tensor, weights: &Tensor) -> Duration {
    let mut duration = Duration::new(0, 0);
    for _i in 0..iters {
        let start = Instant::now();
        let _ = input.matmul(weights);
        duration = duration.checked_add(start.elapsed()).unwrap();
    }
    duration
}

fn bench_tensor_ops(c: &mut Criterion) {
    //    Set-up summarization model
    unsafe {
        torch_sys::dummy_cuda_dependency();
    }
    let input = Tensor::rand(&[32, 128, 512], (Kind::Float, Device::cuda_if_available()));
    let weights = Tensor::rand(&[512, 512], (Kind::Float, Device::cuda_if_available()));

    let _ = &input.matmul(&weights);
    c.bench_function("Matrix multiply ", |b| {
        b.iter_custom(|iters| black_box(matrix_multiply(iters, &input, &weights)))
    });
}

criterion_group! {
name = benches;
config = Criterion::default().sample_size(100);
targets = bench_tensor_ops
}

criterion_main!(benches);
