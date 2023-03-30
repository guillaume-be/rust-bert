use crate::RustBertError;
use ndarray::IxDyn;
use ort::tensor::{DynOrtTensor, FromArray, InputTensor};
use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use tch::{Kind, Tensor};

pub(crate) fn ort_tensor_to_tch(ort_tensor: &DynOrtTensor<IxDyn>) -> Result<Tensor, RustBertError> {
    let ort_tensor = ort_tensor.try_extract::<f32>()?.view().to_owned();
    Ok(Tensor::try_from(ort_tensor)?)
}

pub(crate) fn tch_tensor_to_ort(tch_tensor: &Tensor) -> Result<InputTensor, RustBertError> {
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
/// Container used to store key-value cached states for efficient decoding.
pub struct ONNXLayerCache {
    pub values: HashMap<String, Tensor>,
}

impl ONNXLayerCache {
    /// Helper function to create a cache layer from an ONNX model output.
    /// Assumes that the output names for cached keys and values contain `key` and `value` in their name, respectively.
    pub fn from_ort_output(
        ort_output: &[DynOrtTensor<IxDyn>],
        key_value_names: &HashMap<String, usize>,
    ) -> Result<ONNXLayerCache, RustBertError> {
        let values = key_value_names
            .iter()
            .filter(|(name, _)| name.contains("key") | name.contains("value"))
            .map(|(name, pos)| {
                let value = &ort_output[*pos];
                Ok((name.to_string(), ort_tensor_to_tch(value)?))
            })
            .collect::<Result<HashMap<String, Tensor>, RustBertError>>()?;

        Ok(ONNXLayerCache { values })
    }
}
