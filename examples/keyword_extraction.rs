extern crate anyhow;

use rust_bert::pipelines::keywords_extraction::{
    KeywordExtractionConfig, KeywordExtractionModel, KeywordScorerType,
};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsConfig, SentenceEmbeddingsModelType,
};

fn main() -> anyhow::Result<()> {
    let keyword_extraction_config = KeywordExtractionConfig {
        sentence_embeddings_config: SentenceEmbeddingsConfig::from(
            SentenceEmbeddingsModelType::AllMiniLmL6V2,
        ),
        scorer_type: KeywordScorerType::MaxSum,
        ngram_range: (1, 1),
        num_keywords: 5,
        ..Default::default()
    };

    let keyword_extraction_model = KeywordExtractionModel::new(keyword_extraction_config)?;

    let input = "Rust is a multi-paradigm, general-purpose programming language. \
    Rust emphasizes performance, type safety, and concurrency. Rust enforces memory safety—that is, \
    that all references point to valid memory—without requiring the use of a garbage collector or \
    reference counting present in other memory-safe languages. To simultaneously enforce \
    memory safety and prevent concurrent data races, Rust's borrow checker tracks the object lifetime \
    and variable scope of all references in a program during compilation. Rust is popular for \
    systems programming but also offers high-level features including functional programming constructs.";

    // Credits: Wikimedia https://en.wikipedia.org/wiki/Rust_(programming_language)

    let keywords = keyword_extraction_model.predict(&[input])?;
    for keyword_list in keywords {
        for keyword in keyword_list {
            println!("{:?}, {:?}", keyword.text, keyword.score);
        }
    }
    Ok(())
}
