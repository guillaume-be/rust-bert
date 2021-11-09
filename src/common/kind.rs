use crate::RustBertError;
use tch::{Kind, Scalar};

pub(crate) fn get_positive_infinity(kind: Kind) -> Result<Scalar, RustBertError> {
    Ok(match kind {
        Kind::Uint8 => Scalar::int(u8::MAX.into()),
        Kind::Int8 => Scalar::int(i8::MAX.into()),
        Kind::Int16 => Scalar::int(i16::MAX.into()),
        Kind::Int => Scalar::int(i32::MAX.into()),
        Kind::Int64 => Scalar::int(i64::MAX),
        Kind::Half => Scalar::float(half::f16::INFINITY.into()),
        Kind::Float => Scalar::float(f32::INFINITY.into()),
        Kind::BFloat16 => Scalar::float(half::bf16::INFINITY.into()),
        Kind::Double => Scalar::float(f64::INFINITY),
        _ => {
            return Err(RustBertError::ValueError(format!(
                "Type not supported: attempted to get positive infinity for {:?}",
                kind
            )))
        }
    })
}

pub(crate) fn get_negative_infinity(kind: Kind) -> Result<Scalar, RustBertError> {
    Ok(match kind {
        Kind::Uint8 => Scalar::int(u8::MIN.into()),
        Kind::Int8 => Scalar::int(i8::MIN.into()),
        Kind::Int16 => Scalar::int(i16::MIN.into()),
        Kind::Int => Scalar::int(i32::MIN.into()),
        Kind::Int64 => Scalar::int(i64::MIN),
        Kind::Half => Scalar::float(half::f16::NEG_INFINITY.into()),
        Kind::Float => Scalar::float(f32::NEG_INFINITY.into()),
        Kind::BFloat16 => Scalar::float(half::bf16::NEG_INFINITY.into()),
        Kind::Double => Scalar::float(f64::NEG_INFINITY),
        _ => {
            return Err(RustBertError::ValueError(format!(
                "Type not supported: attempted to get negative infinity for {:?}",
                kind
            )))
        }
    })
}
