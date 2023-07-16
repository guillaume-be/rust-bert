use crate::RustBertError;
use ndarray::{ArrayBase, ArrayD, CowArray, CowRepr, IxDyn};

use ort::{Session, Value};
use std::convert::{TryFrom, TryInto};
use tch::{Kind, Tensor};

pub(crate) fn ort_tensor_to_tch(ort_tensor: &Value) -> Result<Tensor, RustBertError> {
    let ort_tensor = ort_tensor.try_extract::<f32>()?.view().to_owned();
    Ok(Tensor::try_from(ort_tensor)?)
}

pub(crate) fn array_to_ort<'a>(
    session: &Session,
    array: &'a TypedArray<'a>,
) -> Result<Value<'a>, RustBertError> {
    match &array {
        TypedArray::I64(array) => Ok(Value::from_array(session.allocator(), array)?),
        TypedArray::F32(array) => Ok(Value::from_array(session.allocator(), array)?),
        TypedArray::I32(array) => Ok(Value::from_array(session.allocator(), array)?),
        TypedArray::F64(array) => Ok(Value::from_array(session.allocator(), array)?),
        TypedArray::F16(array) => Ok(Value::from_array(session.allocator(), array)?),
        TypedArray::I16(array) => Ok(Value::from_array(session.allocator(), array)?),
        TypedArray::I8(array) => Ok(Value::from_array(session.allocator(), array)?),
        TypedArray::UI8(array) => Ok(Value::from_array(session.allocator(), array)?),
        TypedArray::BF16(array) => Ok(Value::from_array(session.allocator(), array)?),
    }
}

pub(crate) enum TypedArray<'a> {
    I64(ArrayBase<CowRepr<'a, i64>, IxDyn>),
    F32(ArrayBase<CowRepr<'a, f32>, IxDyn>),
    I32(ArrayBase<CowRepr<'a, i32>, IxDyn>),
    F64(ArrayBase<CowRepr<'a, f64>, IxDyn>),
    F16(ArrayBase<CowRepr<'a, half::f16>, IxDyn>),
    I16(ArrayBase<CowRepr<'a, i16>, IxDyn>),
    I8(ArrayBase<CowRepr<'a, i8>, IxDyn>),
    UI8(ArrayBase<CowRepr<'a, u8>, IxDyn>),
    BF16(ArrayBase<CowRepr<'a, half::bf16>, IxDyn>),
}

pub(crate) fn tch_tensor_to_ndarray(tch_tensor: &Tensor) -> Result<TypedArray, RustBertError> {
    let kind = tch_tensor.kind();
    Ok(match kind {
        Kind::Int64 => {
            let array: ArrayD<i64> = tch_tensor.try_into()?;
            TypedArray::I64(CowArray::from(array))
        }
        Kind::Float => {
            let array: ArrayD<f32> = tch_tensor.try_into()?;
            TypedArray::F32(CowArray::from(array))
        }
        Kind::Int => {
            let array: ArrayD<i32> = tch_tensor.try_into()?;
            TypedArray::I32(CowArray::from(array))
        }
        Kind::Double => {
            let array: ArrayD<f64> = tch_tensor.try_into()?;
            TypedArray::F64(CowArray::from(array))
        }
        Kind::Half => {
            let array: ArrayD<half::f16> = tch_tensor.try_into()?;
            TypedArray::F16(CowArray::from(array))
        }
        Kind::Int16 => {
            let array: ArrayD<i16> = tch_tensor.try_into()?;
            TypedArray::I16(CowArray::from(array))
        }
        Kind::Int8 => {
            let array: ArrayD<i8> = tch_tensor.try_into()?;
            TypedArray::I8(CowArray::from(array))
        }
        Kind::Uint8 => {
            let array: ArrayD<u8> = tch_tensor.try_into()?;
            TypedArray::UI8(CowArray::from(array))
        }
        Kind::BFloat16 => {
            let array: ArrayD<half::bf16> = tch_tensor.try_into()?;
            TypedArray::BF16(CowArray::from(array))
        }
        _ => {
            return Err(RustBertError::ValueError(format!(
                "Type not supported: attempted to get convert torch tensor to ndarray for {kind:?}",
            )))
        }
    })
}
