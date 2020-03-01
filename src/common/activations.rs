use tch::Tensor;
use std::f64::consts::PI;

pub fn _gelu(x: &Tensor) -> Tensor { x * 0.5 * (1.0 + (x / ((2.0 as f64).sqrt())).erf()) }

pub fn _relu(x: &Tensor) -> Tensor { x.relu() }

pub fn _swish(x: &Tensor) -> Tensor {x * x.sigmoid()}

pub fn _mish(x: &Tensor) -> Tensor { x * (x.softplus().tanh()) }

pub fn _gelu_new(x: &Tensor) -> Tensor { x * 0.5 * (((x.pow(3.0f64) * 0.044715 + x) * ((2f64 / PI).sqrt())).tanh() + 1) }