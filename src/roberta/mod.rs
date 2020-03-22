mod embeddings;
mod roberta;

pub use roberta::{RobertaForMaskedLM, RobertaForMultipleChoice, RobertaForTokenClassification, RobertaForQuestionAnswering, RobertaForSequenceClassification};
pub use embeddings::RobertaEmbeddings;