#[cfg(feature = "onnx")]
use ndarray::ShapeError;
#[cfg(feature = "onnx")]
use ort::OrtError;
use rust_tokenizers::error::TokenizerError;
use tch::TchError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RustBertError {
    #[cfg(feature = "remote")]
    #[error("Endpoint not available error: {0}")]
    FileDownloadError(#[from] cached_path::Error),

    #[error("IO error: {0}")]
    IOError(String),

    #[error("Tch tensor error: {0}")]
    TchError(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Invalid configuration error: {0}")]
    InvalidConfigurationError(String),

    #[error("Value error: {0}")]
    ValueError(String),

    #[error("Value error: {0}")]
    #[cfg(feature = "onnx")]
    OrtError(String),

    #[error("Value error: {0}")]
    #[cfg(feature = "onnx")]
    NdArrayError(String),

    #[error("Unsupported operation")]
    UnsupportedError,
}

impl From<std::io::Error> for RustBertError {
    fn from(error: std::io::Error) -> Self {
        RustBertError::IOError(error.to_string())
    }
}

impl From<TokenizerError> for RustBertError {
    fn from(error: TokenizerError) -> Self {
        RustBertError::TokenizerError(error.to_string())
    }
}

impl From<TchError> for RustBertError {
    fn from(error: TchError) -> Self {
        RustBertError::TchError(error.to_string())
    }
}

#[cfg(feature = "onnx")]
impl From<OrtError> for RustBertError {
    fn from(error: OrtError) -> Self {
        RustBertError::OrtError(error.to_string())
    }
}
#[cfg(feature = "onnx")]
impl From<ShapeError> for RustBertError {
    fn from(error: ShapeError) -> Self {
        RustBertError::NdArrayError(error.to_string())
    }
}
