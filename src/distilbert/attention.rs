use crate::distilbert::dropout::Dropout;
use tch::{nn, Tensor};
use crate::distilbert::distilbert::DistilBertConfig;

#[derive(Debug)]
pub struct MultiHeadSelfAttention {
    n_heads: i64,
    dim: i64,
    dropout: Dropout,
    output_attentions: bool,
    q_lin: nn::Linear,
    k_lin: nn::Linear,
    v_lin: nn::Linear,
    out_lin: nn::Linear,
}

impl MultiHeadSelfAttention {
    pub fn new(p: nn::Path, config: &DistilBertConfig) -> MultiHeadSelfAttention {
        let q_lin = nn::linear(&p / "q_lin", config.dim, config.dim, Default::default());
        let k_lin = nn::linear(&p / "k_lin", config.dim, config.dim, Default::default());
        let v_lin = nn::linear(&p / "v_lin", config.dim, config.dim, Default::default());
        let out_lin = nn::linear(&p / "out_lin", config.dim, config.dim, Default::default());

        let dropout = Dropout::new(config.attention_dropout);

        MultiHeadSelfAttention {
            n_heads: config.n_heads,
            dim: config.dim,
            dropout,
            output_attentions: config.output_attentions,
            q_lin,
            k_lin,
            v_lin,
            out_lin,
        }
    }

    fn shape(&self, x: Tensor, bs: i64, dim_per_head: i64) -> Tensor {
        x.view((bs, -1, self.n_heads, dim_per_head)).transpose(1, 2)
    }

    fn unshape(&self, x: Tensor, bs: i64, dim_per_head: i64) -> Tensor {
        x.transpose(1, 2).contiguous().view((bs, -1, &self.n_heads * dim_per_head))
    }

    fn forward_t(&self, query: &Tensor, key: &Tensor, value: &Tensor, mask: &Tensor, train: bool) -> Tensor {
        let input_size = query.size();
        let (bs, q_length, dim) = (input_size[0], input_size[1], input_size[2]);
        let k_length = key.size()[1];
        let dim_per_head = self.dim / self.n_heads;
        let mask_reshape = (bs, 1i64, 1i64, k_length);

        let q = self.shape(query.apply(&self.q_lin), bs, dim_per_head);
        let k = self.shape(key.apply(&self.k_lin), bs, dim_per_head);
        let v = self.shape(value.apply(&self.v_lin), bs, dim_per_head);
        let q: Tensor = q / (dim_per_head as f64).sqrt();

        let scores = q.matmul(&k.transpose(2, 3));
//    ToDo: add masking calculation


        Tensor::new()
    }
}
