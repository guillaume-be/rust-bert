use crate::pipelines::keywords_extraction::KeywordScorerType;
use tch::{Kind, Tensor};

impl KeywordScorerType {
    pub fn score_keywords(
        &self,
        document_embedding: Tensor,
        word_embeddings: Tensor,
        num_keywords: usize,
        diversity: Option<f32>,
        max_sum_candidates: Option<usize>,
    ) -> Vec<usize> {
        match self {
            KeywordScorerType::CosineSimilarity => {
                cosine_similarity(document_embedding, word_embeddings, num_keywords)
            }
            KeywordScorerType::MaximalMarginRelevance => {
                todo!();
            }
            KeywordScorerType::MaxSum => {
                todo!();
            }
        }
    }
}

fn cosine_similarity(
    document_embedding: Tensor,
    word_embeddings: Tensor,
    num_keywords: usize,
) -> Vec<usize> {
    let document_embedding = &document_embedding
        / document_embedding.linalg_norm(2.0, vec![0i64].as_slice(), true, Kind::Float);
    let word_embeddings = &word_embeddings
        / word_embeddings.linalg_norm(2.0, vec![1i64].as_slice(), true, Kind::Float);

    let similarities = document_embedding.matmul(&word_embeddings.transpose(0, 1));
    let (_, top_keywords) = similarities.topk(num_keywords as i64, 0, true, false);
    top_keywords.print();
    top_keywords
        .iter::<i64>()
        .unwrap()
        .map(|pos| pos as usize)
        .collect::<Vec<usize>>()
}
