use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use tch::Tensor;

pub fn _gelu(x: &Tensor) -> Tensor {
    x * 0.5 * (1.0 + (x / ((2.0 as f64).sqrt())).erf())
}

pub fn _relu(x: &Tensor) -> Tensor {
    x.relu()
}

pub fn _swish(x: &Tensor) -> Tensor {
    x * x.sigmoid()
}

pub fn _mish(x: &Tensor) -> Tensor {
    x * (x.softplus().tanh())
}

pub fn _gelu_new(x: &Tensor) -> Tensor {
    x * 0.5 * (((x.pow(3.0f64) * 0.044715 + x) * ((2f64 / PI).sqrt())).tanh() + 1)
}

pub fn _tanh(x: &Tensor) -> Tensor {
    x.tanh()
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize)]
/// # Activation function used in the attention layer and masked language model head
pub enum Activation {
    /// Gaussian Error Linear Unit ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu,
    /// Rectified Linear Unit
    relu,
    /// Swish ([Ramachandran, 2017](https://arxiv.org/abs/1710.05941))
    swish,
    /// Mish ([Misra, 2019](https://arxiv.org/abs/1908.08681))
    mish,
    /// Gaussian Error Linear Unit (New) ([Hendrycks et al., 2016,](https://arxiv.org/abs/1606.08415))
    gelu_new,
    /// Tanh
    tanh,
}
