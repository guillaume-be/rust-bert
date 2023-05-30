use crate::RustBertError;
use ndarray::IxDyn;
use ort::tensor::{DynOrtTensor, FromArray, InputTensor};
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
