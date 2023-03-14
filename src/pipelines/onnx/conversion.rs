use crate::RustBertError;
use ndarray::{Dimension, IxDyn};
use ort::tensor::DynOrtTensor;
use ort::{OrtApiError, OrtError};
use std::collections::HashMap;
use tch::Tensor;

pub fn ort_tensor_to_tch(ort_tensor: &DynOrtTensor<IxDyn>) -> Result<Tensor, RustBertError> {
    let ort_tensor = ort_tensor.try_extract::<f32>()?;
    let dim = ort_tensor
        .view()
        .dim()
        .as_array_view()
        .iter()
        .map(|dim| *dim as i64)
        .collect::<Vec<_>>();
    Ok(
        Tensor::of_slice(ort_tensor.view().as_slice().ok_or_else(|| {
            return OrtError::FailedTensorCheck(OrtApiError::Msg(
                "Non-contiguous tensor encountered during conversion to tch".to_string(),
            ));
        })?)
        .view(dim.as_slice()),
    )
}

#[derive(Debug)]
pub struct ONNXLayerCache {
    pub values: HashMap<String, Tensor>,
}

impl ONNXLayerCache {
    pub fn from_ort_output(
        ort_output: &'_ Vec<DynOrtTensor<IxDyn>>,
        key_value_names: &HashMap<String, usize>,
    ) -> Result<ONNXLayerCache, RustBertError> {
        let values = key_value_names
            .iter()
            .filter(|(name, _)| name.contains(".key") | name.contains(".value"))
            .map(|(name, pos)| {
                let value = &ort_output[*pos];
                Ok((name.to_string(), ort_tensor_to_tch(value)?))
            })
            .collect::<Result<HashMap<String, Tensor>, RustBertError>>()?;

        Ok(ONNXLayerCache { values })
    }
}
