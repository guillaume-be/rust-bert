use tch::{nn, Tensor, Kind};
use tch::nn::{ModuleT, embedding};
use crate::distilbert::distilbert::DistilBertConfig;

//fn create_sinusoidal_embeddings() {}

pub fn embeddings(p: nn::Path, config: DistilBertConfig) -> impl ModuleT {
    let word_embeddings: nn::Embedding = embedding(&p / "word_embeddings",
                                                   config.vocab_size,
                                                   config.dim,
                                                   Default::default());
    let position_embeddings: nn::Embedding = embedding(&p / "position_embeddings",
                                                       config.max_position_embeddings,
                                                       config.dim,
                                                       Default::default());

    nn::func_t(move |input_ids, _train| {
        let seq_length = (&input_ids).size().last().unwrap().to_owned();
        let position_ids = Tensor::arange(seq_length, (Kind::Int64, input_ids.device()));
        let position_ids = position_ids.unsqueeze(0).expand_as(input_ids);

        let word_embed = input_ids.apply(&word_embeddings);
        let position_embed = position_ids.apply(&position_embeddings);

        let embeddings = word_embed + position_embed;

        embeddings
    })
}