//! # Longformer: The Long-Document Transformer (Betalgy et al.)
//!
//! Implementation of the Longformer language model ([Longformer: The Long-Document Transformer](https://arxiv.org/abs/2001.04063) Betalgy, Peters, Cohan, 2020).
//! The base model is implemented in the `longformer_model::LongformerModel` struct. Several language model heads have also been implemented, including:
//! - Masked language model: `longformer_model::LongformerForMaskedLM`
//! - Multiple choices: `longformer_model:LongformerForMultipleChoice`
//! - Question answering: `longformer_model::LongformerForQuestionAnswering`
//! - Sequence classification: `longformer_model::LongformerForSequenceClassification`
//! - Token classification (e.g. NER, POS tagging): `longformer_model::LongformerForTokenClassification`
//!
//! # Model set-up and pre-trained weights loading
//!
//! A full working example (question answering) is provided in `examples/question_answering_longformer`, run with `cargo run --example question_answering_longformer`.
//! All models expect the following resources:
//! - Configuration file expected to have a structure following the [Transformers library](https://github.com/huggingface/transformers)
//! - Model weights are expected to have a structure and parameter names following the [Transformers library](https://github.com/huggingface/transformers). A conversion using the Python utility scripts is required to convert the `.bin` weights to the `.ot` format.
//! - `RobertaTokenizer` using a `vocab.json` vocabulary and `merges.txt` byte pair encoding merges
//!
//! # Question answering example below:
//!
//! ```no_run
//! use rust_bert::longformer::{
//!    LongformerConfigResources, LongformerMergesResources, LongformerModelResources,
//!    LongformerVocabResources,
//! };
//! use rust_bert::pipelines::common::ModelType;
//! use rust_bert::pipelines::question_answering::{
//!    QaInput, QuestionAnsweringConfig, QuestionAnsweringModel,
//! };
//! use rust_bert::resources::{RemoteResource};
//!
//! fn main() -> anyhow::Result<()> {
//!    //    Set-up Question Answering model
//!    let config = QuestionAnsweringConfig::new(
//!        ModelType::Longformer,
//!        RemoteResource::from_pretrained(
//!            LongformerModelResources::LONGFORMER_BASE_SQUAD1,
//!        ),
//!        RemoteResource::from_pretrained(
//!            LongformerConfigResources::LONGFORMER_BASE_SQUAD1,
//!        ),
//!        RemoteResource::from_pretrained(
//!            LongformerVocabResources::LONGFORMER_BASE_SQUAD1,
//!        ),
//!        Some(RemoteResource::from_pretrained(
//!            LongformerMergesResources::LONGFORMER_BASE_SQUAD1,
//!        )),
//!        false,
//!        None,
//!        false,
//!    );
//!
//!    let qa_model = QuestionAnsweringModel::new(config)?;
//!
//!    //    Define input
//!    let question_1 = String::from("Where does Amy live ?");
//!    let context_1 = String::from("Amy lives in Amsterdam");
//!    let question_2 = String::from("Where does Eric live");
//!    let context_2 = String::from("While Amy lives in Amsterdam, Eric is in The Hague.");
//!    let qa_input_1 = QaInput {
//!        question: question_1,
//!        context: context_1,
//!    };
//!    let qa_input_2 = QaInput {
//!        question: question_2,
//!        context: context_2,
//!    };
//!
//!    //    Get answer
//!    let answers = qa_model.predict(&[qa_input_1, qa_input_2], 1, 32);
//!    println!("{:?}", answers);
//!    Ok(())
//! }
//!  ```

mod attention;
mod embeddings;
mod encoder;
mod longformer_model;

pub use longformer_model::{
    LongformerConfig, LongformerConfigResources, LongformerForMaskedLM,
    LongformerForMultipleChoice, LongformerForQuestionAnswering,
    LongformerForSequenceClassification, LongformerForTokenClassification,
    LongformerMergesResources, LongformerModel, LongformerModelResources,
    LongformerTokenClassificationOutput, LongformerVocabResources,
};
