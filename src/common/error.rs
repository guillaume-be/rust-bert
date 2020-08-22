use rust_tokenizers::preprocessing::error::TokenizerError;
use tch::TchError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RustBertError {
    #[error("File not found error: {0}")]
    FileNotFound(String),

    #[error("Tch tensor error: {0}")]
    TchError(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
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
