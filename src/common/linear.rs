use tch::nn::{Init, Path, Module};
use tch::Tensor;
use std::borrow::Borrow;

#[derive(Debug, Clone, Copy)]
pub struct LinearNoBiasConfig {
    pub ws_init: Init,
}

impl Default for LinearNoBiasConfig {
    fn default() -> Self {
        LinearNoBiasConfig {
            ws_init: Init::KaimingUniform,
        }
    }
}

#[derive(Debug)]
pub struct LinearNoBias {
    pub ws: Tensor,
}


pub fn linear_no_bias<'a, T: Borrow<Path<'a>>>(
    vs: T,
    in_dim: i64,
    out_dim: i64,
    c: LinearNoBiasConfig,
) -> LinearNoBias {
    let vs = vs.borrow();
    LinearNoBias {
        ws: vs.var("weight", &[out_dim, in_dim], c.ws_init),
    }
}

impl Module for LinearNoBias {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.matmul(&self.ws.tr())
    }
}