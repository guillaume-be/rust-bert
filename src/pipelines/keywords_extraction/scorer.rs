/// Derived from https://github.com/MaartenGr/KeyBERT, shared under MIT License
///
/// Copyright (c) 2020, Maarten P. Grootendorst
/// Copyright (c) 2022, Guillaume Becquin
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in all
/// copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
/// SOFTWARE.
use crate::pipelines::keywords_extraction::KeywordScorerType;
use std::cmp::{max, min};
use std::convert::TryFrom;
use tch::{Kind, Tensor};

impl KeywordScorerType {
    pub(crate) fn score_keywords(
        &self,
        document_embedding: Tensor,
        word_embeddings: Tensor,
        num_keywords: usize,
        diversity: Option<f64>,
        max_sum_candidates: Option<usize>,
    ) -> Vec<(usize, f32)> {
        match self {
            KeywordScorerType::CosineSimilarity => {
                cosine_similarity_score(document_embedding, word_embeddings, num_keywords)
            }
            KeywordScorerType::MaximalMarginRelevance => maximal_margin_relevance_score(
                document_embedding,
                word_embeddings,
                num_keywords,
                diversity.unwrap_or(0.5),
            ),
            KeywordScorerType::MaxSum => {
                let num_keywords_candidates = word_embeddings.size()[0] as usize;
                max_sum_score(
                    document_embedding,
                    word_embeddings,
                    num_keywords,
                    min(
                        max_sum_candidates.unwrap_or(num_keywords * 2),
                        num_keywords_candidates,
                    ),
                )
            }
        }
    }
}

fn cosine_similarity(document_embedding: Option<&Tensor>, word_embeddings: &Tensor) -> Tensor {
    let word_embeddings = word_embeddings
        / word_embeddings.linalg_norm(2.0, vec![1i64].as_slice(), true, Kind::Float);
    let reference_embedding = document_embedding.map(|embedding| {
        embedding / embedding.linalg_norm(2.0, vec![1i64].as_slice(), true, Kind::Float)
    });
    let reference_embedding = reference_embedding.as_ref().unwrap_or(&word_embeddings);

    reference_embedding.matmul(&word_embeddings.transpose(0, 1))
}

fn cosine_similarity_score(
    document_embedding: Tensor,
    word_embeddings: Tensor,
    num_keywords: usize,
) -> Vec<(usize, f32)> {
    let similarities = cosine_similarity(Some(&document_embedding), &word_embeddings).view([-1]);

    let (top_scores, top_keywords) = similarities.topk(num_keywords as i64, 0, true, false);
    top_scores
        .iter::<f64>()
        .unwrap()
        .zip(top_keywords.iter::<i64>().unwrap())
        .map(|(score, pos)| (pos as usize, score as f32))
        .collect()
}

fn maximal_margin_relevance_score(
    document_embedding: Tensor,
    word_embeddings: Tensor,
    num_keywords: usize,
    diversity: f64,
) -> Vec<(usize, f32)> {
    let word_document_similarities =
        cosine_similarity(Some(&document_embedding), &word_embeddings).view([-1]);
    let word_similarities = cosine_similarity(None, &word_embeddings);

    let mut keyword_indices =
        vec![i64::try_from(word_document_similarities.argmax(0, false)).unwrap()];
    let mut candidate_indices = (0..word_document_similarities.size()[0]).collect::<Vec<i64>>();
    let _ = candidate_indices.remove(keyword_indices[0] as usize);
    for _ in 0..min(num_keywords - 1, word_embeddings.size()[0] as usize) {
        let candidate_indices_tensor =
            Tensor::from_slice(&candidate_indices).to(word_document_similarities.device());
        let candidate_similarities =
            word_document_similarities.index_select(0, &candidate_indices_tensor);
        let (target_similarities, _) = word_similarities
            .index_select(0, &candidate_indices_tensor)
            .index_select(
                1,
                &Tensor::from_slice(&keyword_indices).to(word_similarities.device()),
            )
            .max_dim(1, false);
        let mmr = candidate_similarities * (1.0 - diversity) - target_similarities * diversity;
        let mmr_index = candidate_indices[i64::try_from(mmr.argmax(0, false)).unwrap() as usize];
        keyword_indices.push(mmr_index);
        let candidate_mmr_index = candidate_indices
            .iter()
            .position(|x| *x == mmr_index)
            .unwrap();
        candidate_indices.remove(candidate_mmr_index);
    }

    keyword_indices
        .into_iter()
        .map(|index| {
            (
                index as usize,
                word_document_similarities.double_value(&[index]) as f32,
            )
        })
        .collect()
}

fn max_sum_score(
    document_embedding: Tensor,
    word_embeddings: Tensor,
    num_keywords: usize,
    max_sum_candidates: usize,
) -> Vec<(usize, f32)> {
    let max_sum_candidates = max(num_keywords, max_sum_candidates);
    let word_document_similarities =
        cosine_similarity(Some(&document_embedding), &word_embeddings).view([-1]);
    let word_similarities = cosine_similarity(None, &word_embeddings);
    let (_, top_keywords) =
        word_document_similarities.topk(max_sum_candidates as i64, 0, true, false);

    let keyword_combinations = top_keywords.combinations(num_keywords as i64, false);
    let (mut best_score, mut best_combination) = (None, None);
    for idx in 0..keyword_combinations.size()[0] {
        let combination = keyword_combinations.get(idx);
        let combination_score = f64::try_from(
            word_similarities
                .index_select(0, &combination)
                .index_select(1, &combination)
                .sum(word_similarities.kind()),
        )
        .unwrap();
        if let Some(current_best_score) = best_score {
            if combination_score < current_best_score {
                best_score = Some(combination_score);
                best_combination = Some(combination);
            }
        } else {
            best_score = Some(combination_score);
            best_combination = Some(combination);
        }
    }

    best_combination
        .unwrap()
        .iter::<i64>()
        .unwrap()
        .map(|index| {
            (
                index as usize,
                word_document_similarities.double_value(&[index]) as f32,
            )
        })
        .collect()
}
