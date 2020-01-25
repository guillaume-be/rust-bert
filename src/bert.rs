extern crate tch;

use tch::Tensor;

fn gelu(x: Tensor) -> Tensor {
    &x * 0.5 * (1.0 + (x / ((2.0 as f64).sqrt())).erf())
}