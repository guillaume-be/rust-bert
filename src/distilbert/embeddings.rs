use tch::{nn, Tensor, Kind, Device};
use tch::nn::{ModuleT, embedding, layer_norm, EmbeddingConfig};
use crate::distilbert::distilbert::DistilBertConfig;
use crate::distilbert::dropout::Dropout;


fn create_sinusoidal_embeddings(config: &DistilBertConfig, device: Device) -> nn::Embedding {
    let sinusoidal_embedding = Tensor::arange(config.max_position_embeddings, (Kind::Float, device)).unsqueeze(1);
    let multiplier: Tensor = Tensor::arange2(0, config.dim, 2, (Kind::Float, device));
    let multiplier: Tensor = Tensor::from(1.0) / (Tensor::ones(&[1], (Kind::Float, device)) * 10000).pow1(&(multiplier / config.dim));
    let sinusoidal_embedding: Tensor = sinusoidal_embedding * multiplier;
    let cos_embeddings: Tensor = sinusoidal_embedding.cos();
    let sin_embeddings: Tensor = sinusoidal_embedding.sin();

    let sinusoidal_embedding: Tensor = Tensor::ones(&[config.max_position_embeddings, config.dim], (Kind::Float, device));
    sinusoidal_embedding.slice(1, 0, config.dim, 2).copy_(&sin_embeddings);
    sinusoidal_embedding.slice(1, 1, config.dim, 2).copy_(&cos_embeddings);

    let embedding_config = EmbeddingConfig { padding_idx: 0, ..Default::default() };
    let mut embeddings = embedding(&nn::VarStore::new(device).root(),
                                   config.max_position_embeddings,
                                   config.dim,
                                   embedding_config);
    embeddings.ws = sinusoidal_embedding;
    embeddings
}


#[derive(Debug)]
pub struct BertEmbedding {
    word_embeddings: nn::Embedding,
    position_embeddings: nn::Embedding,
    layer_norm: nn::LayerNorm,
    dropout: Dropout,
}

impl BertEmbedding {
    pub fn new(p: nn::Path, config: &DistilBertConfig) -> BertEmbedding {

        let embedding_config = EmbeddingConfig { padding_idx: 0, ..Default::default() };

        let word_embeddings: nn::Embedding = embedding(&p / "word_embeddings",
                                                       config.vocab_size,
                                                       config.dim,
                                                       embedding_config);
        let position_embeddings: nn::Embedding = match config.sinusoidal_pos_embds {
            false => embedding(&p / "position_embeddings",
                               config.max_position_embeddings,
                               config.dim,
                               embedding_config),

            true => create_sinusoidal_embeddings(&config, p.device())
        };
        let layer_norm: nn::LayerNorm = layer_norm(&p, vec![config.dim], Default::default());
        let dropout: Dropout = Dropout::new(config.dropout);
        BertEmbedding { word_embeddings, position_embeddings, layer_norm, dropout }
    }
}

impl ModuleT for BertEmbedding {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        let seq_length = (&input).size().last().unwrap().to_owned();
        let position_ids = Tensor::arange(seq_length, (Kind::Int64, input.device()));
        let position_ids = position_ids.unsqueeze(0).expand_as(input);

        let word_embed = input.apply(&self.word_embeddings);
        let position_embed = position_ids.apply(&self.position_embeddings);

        let embeddings = word_embed + position_embed;
        let embeddings = embeddings.apply(&self.layer_norm).apply_t(&self.dropout, train);

        embeddings
    }
}