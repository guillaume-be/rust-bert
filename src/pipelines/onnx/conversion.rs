use crate::RustBertError;
use ndarray::{Dimension, IxDyn};
use ort::tensor::{DynOrtTensor, FromArray, InputTensor};
use ort::{OrtApiError, OrtError};
use std::collections::HashMap;
use std::convert::TryInto;
use tch::{Kind, Tensor};

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
            OrtError::FailedTensorCheck(OrtApiError::Msg(
                "Non-contiguous tensor encountered during conversion to tch".to_string(),
            ))
        })?)
        .view(dim.as_slice()),
    )
}

pub fn tch_tensor_to_ort(tch_tensor: &Tensor) -> Result<InputTensor, RustBertError> {
    let kind = tch_tensor.kind();
    Ok(match kind{
        Kind::Int64 => {
            let array: ndarray::ArrayD<i64> = tch_tensor.try_into()?;
            InputTensor::from_array(array)
        }
        Kind::Float => {
            let array: ndarray::ArrayD<f32> = tch_tensor.try_into()?;
            InputTensor::from_array(array)
        }
        Kind::Int => {
            let array: ndarray::ArrayD<i32> = tch_tensor.try_into()?;
            InputTensor::from_array(array)
        }
        Kind::Double => {
            let array: ndarray::ArrayD<f64> = tch_tensor.try_into()?;
            InputTensor::from_array(array)
        }
        Kind::Half => {
            let array: ndarray::ArrayD<half::f16> = tch_tensor.try_into()?;
            InputTensor::from_array(array)
        }
        Kind::Int16 => {
            let array: ndarray::ArrayD<i16> = tch_tensor.try_into()?;
            InputTensor::from_array(array)
        }
        Kind::Int8 => {
            let array: ndarray::ArrayD<i8> = tch_tensor.try_into()?;
            InputTensor::from_array(array)
        }
        Kind::Uint8 => {
            let array: ndarray::ArrayD<u8> = tch_tensor.try_into()?;
            InputTensor::from_array(array)
        }
        Kind::BFloat16 => {
            let array: ndarray::ArrayD<half::bf16> = tch_tensor.try_into()?;
            InputTensor::from_array(array)
        }
        _ => {
            return Err(RustBertError::ValueError(format!(
                "Type not supported: attempted to get convert torch tensor to ndarray infinity for {kind:?}",
            )))
        }
    })
}

#[derive(Debug)]
pub struct ONNXLayerCache {
    pub values: HashMap<String, Tensor>,
}

impl ONNXLayerCache {
    pub fn from_ort_output(
        ort_output: &[DynOrtTensor<IxDyn>],
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
