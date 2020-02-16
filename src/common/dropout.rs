use tch::nn::ModuleT;
use tch::Tensor;

#[derive(Debug)]
pub struct Dropout {
    p: f64,
}

impl Dropout {
    pub fn new(p: f64) -> Dropout {
        Dropout { p }
    }
}

impl ModuleT for Dropout {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        input.dropout(self.p, train)
    }
}