use crate::deberta::BaseDebertaEmbeddings;
use tch::nn::LayerNorm;

pub type DebertaV2Embeddings = BaseDebertaEmbeddings<LayerNorm>;
