use crate::RustBertError;
use tch::nn::Embedding;
use tch::{Device, Tensor};

pub fn process_ids_embeddings_pair(
    input_ids: Option<&Tensor>,
    input_embeddings: Option<&Tensor>,
    embeddings_matrix: &Embedding,
) -> Result<(Option<Tensor>, Vec<i64>, Device), RustBertError> {
    Ok(match (input_ids, input_embeddings) {
        (Some(_), Some(_)) => {
            return Err(RustBertError::ValueError(
                "Only one of input ids or input embeddings may be set".into(),
            ));
        }
        (Some(input_value), None) => (
            Some(input_value.apply(embeddings_matrix)),
            input_value.size(),
            input_value.device(),
        ),
        (None, Some(embeds)) => {
            let size = vec![embeds.size()[0], embeds.size()[1]];
            (None, size, embeds.device())
        }
        (None, None) => {
            return Err(RustBertError::ValueError(
                "At least one of input ids or input embeddings must be set".into(),
            ));
        }
    })
}

pub fn get_shape_and_device_from_ids_embeddings_pair(
    input_ids: Option<&Tensor>,
    input_embeddings: Option<&Tensor>,
) -> Result<(Vec<i64>, Device), RustBertError> {
    Ok(match (input_ids, input_embeddings) {
        (Some(_), Some(_)) => {
            return Err(RustBertError::ValueError(
                "Only one of input ids or input embeddings may be set".into(),
            ));
        }
        (Some(input_value), None) => (input_value.size(), input_value.device()),
        (None, Some(embeds)) => {
            let size = vec![embeds.size()[0], embeds.size()[1]];
            (size, embeds.device())
        }
        (None, None) => {
            return Err(RustBertError::ValueError(
                "At least one of input ids or input embeddings must be set".into(),
            ));
        }
    })
}
