use tch::Tensor;

pub fn _gelu(x: &Tensor) -> Tensor { x * 0.5 * (1.0 + (x / ((2.0 as f64).sqrt())).erf()) }

pub fn _relu(x: &Tensor) -> Tensor { x.relu() }

pub fn _mish(x: &Tensor) -> Tensor { x * (x.softplus().tanh()) }

