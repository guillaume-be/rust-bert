use rust_bert::pipelines::keywords_extraction::{
    KeywordExtractionConfig, KeywordExtractionModel, KeywordScorerType,
};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsConfig, SentenceEmbeddingsModelType,
};

#[test]
fn sbert_distilbert() -> anyhow::Result<()> {
    let model = SentenceEmbeddingsBuilder::remote(
        SentenceEmbeddingsModelType::DistiluseBaseMultilingualCased,
    )
    .create_model()?;

    let sentences = ["This is an example sentence", "Each sentence is converted"];
    let embeddings = model.encode(&sentences)?;

    assert!((embeddings[0][0] as f64 - -0.03479306).abs() < 1e-4);
    assert!((embeddings[0][1] as f64 - 0.02635195).abs() < 1e-4);
    assert!((embeddings[0][2] as f64 - -0.04427199).abs() < 1e-4);
    assert!((embeddings[0][509] as f64 - 0.01743882).abs() < 1e-4);
    assert!((embeddings[0][510] as f64 - -0.01952395).abs() < 1e-4);
    assert!((embeddings[0][511] as f64 - -0.00118101).abs() < 1e-4);

    assert!((embeddings[1][0] as f64 - 0.02096637).abs() < 1e-4);
    assert!((embeddings[1][1] as f64 - -0.00401743).abs() < 1e-4);
    assert!((embeddings[1][2] as f64 - -0.05093712).abs() < 1e-4);
    assert!((embeddings[1][509] as f64 - 0.03618195).abs() < 1e-4);
    assert!((embeddings[1][510] as f64 - 0.0294408).abs() < 1e-4);
    assert!((embeddings[1][511] as f64 - -0.04497765).abs() < 1e-4);

    Ok(())
}

#[test]
fn sbert_bert() -> anyhow::Result<()> {
    let model =
        SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::BertBaseNliMeanTokens)
            .create_model()?;

    let sentences = ["this is an example sentence", "each sentence is converted"];
    let embeddings = model.encode(&sentences)?;

    assert!((embeddings[0][0] as f64 - -0.393099815).abs() < 1e-4);
    assert!((embeddings[0][1] as f64 - 0.0388629436).abs() < 1e-4);
    assert!((embeddings[0][2] as f64 - 1.98742473).abs() < 1e-4);
    assert!((embeddings[0][765] as f64 - -0.609367728).abs() < 1e-4);
    assert!((embeddings[0][766] as f64 - -1.09462142).abs() < 1e-4);
    assert!((embeddings[0][767] as f64 - 0.326490253).abs() < 1e-4);

    assert!((embeddings[1][0] as f64 - 0.0615336187).abs() < 1e-4);
    assert!((embeddings[1][1] as f64 - 0.32736221).abs() < 1e-4);
    assert!((embeddings[1][2] as f64 - 1.8332324).abs() < 1e-4);
    assert!((embeddings[1][765] as f64 - -0.129853949).abs() < 1e-4);
    assert!((embeddings[1][766] as f64 - 0.460893631).abs() < 1e-4);
    assert!((embeddings[1][767] as f64 - 0.240354523).abs() < 1e-4);

    Ok(())
}

#[test]
fn sbert_bert_small() -> anyhow::Result<()> {
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()?;

    let sentences = ["this is an example sentence", "each sentence is converted"];
    let embeddings = model.encode(&sentences)?;

    assert!((embeddings[0][0] as f64 - -2.02682902e-04).abs() < 1e-4);
    assert!((embeddings[0][1] as f64 - 8.14802647e-02).abs() < 1e-4);
    assert!((embeddings[0][2] as f64 - 3.13617811e-02).abs() < 1e-4);
    assert!((embeddings[0][381] as f64 - 6.20930083e-02).abs() < 1e-4);
    assert!((embeddings[0][382] as f64 - 4.91031967e-02).abs() < 1e-4);
    assert!((embeddings[0][383] as f64 - -2.90199649e-04).abs() < 1e-4);

    assert!((embeddings[1][0] as f64 - 6.47571534e-02).abs() < 1e-4);
    assert!((embeddings[1][1] as f64 - 4.85198125e-02).abs() < 1e-4);
    assert!((embeddings[1][2] as f64 - -1.78603437e-02).abs() < 1e-4);
    assert!((embeddings[1][381] as f64 - 3.37569155e-02).abs() < 1e-4);
    assert!((embeddings[1][382] as f64 - 8.43371451e-03).abs() < 1e-4);
    assert!((embeddings[1][383] as f64 - -6.00359812e-02).abs() < 1e-4);

    Ok(())
}

#[test]
fn sbert_distilroberta() -> anyhow::Result<()> {
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllDistilrobertaV1)
        .create_model()?;

    let sentences = ["This is an example sentence", "Each sentence is converted"];
    let embeddings = model.encode(&sentences)?;

    assert!((embeddings[0][0] as f64 - -0.03375624).abs() < 1e-4);
    assert!((embeddings[0][1] as f64 - -0.06316338).abs() < 1e-4);
    assert!((embeddings[0][2] as f64 - -0.0316612).abs() < 1e-4);
    assert!((embeddings[0][765] as f64 - 0.03684864).abs() < 1e-4);
    assert!((embeddings[0][766] as f64 - -0.02036646).abs() < 1e-4);
    assert!((embeddings[0][767] as f64 - -0.01574).abs() < 1e-4);

    assert!((embeddings[1][0] as f64 - -0.01409588).abs() < 1e-4);
    assert!((embeddings[1][1] as f64 - 0.00091114).abs() < 1e-4);
    assert!((embeddings[1][2] as f64 - -0.00096315).abs() < 1e-4);
    assert!((embeddings[1][765] as f64 - -0.02571585).abs() < 1e-4);
    assert!((embeddings[1][766] as f64 - -0.00289072).abs() < 1e-4);
    assert!((embeddings[1][767] as f64 - -0.00579975).abs() < 1e-4);

    Ok(())
}

#[test]
fn sbert_albert() -> anyhow::Result<()> {
    let model =
        SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::ParaphraseAlbertSmallV2)
            .create_model()?;

    let sentences = ["this is an example sentence", "each sentence is converted"];
    let embeddings = model.encode(&sentences)?;

    assert!((embeddings[0][0] as f64 - 0.20412037).abs() < 1e-4);
    assert!((embeddings[0][1] as f64 - 0.48823047).abs() < 1e-4);
    assert!((embeddings[0][2] as f64 - 0.5664698).abs() < 1e-4);
    assert!((embeddings[0][765] as f64 - -0.37474486).abs() < 1e-4);
    assert!((embeddings[0][766] as f64 - 0.0254627).abs() < 1e-4);
    assert!((embeddings[0][767] as f64 - -0.6846024).abs() < 1e-4);

    assert!((embeddings[1][0] as f64 - 0.25720373).abs() < 1e-4);
    assert!((embeddings[1][1] as f64 - 0.24648172).abs() < 1e-4);
    assert!((embeddings[1][2] as f64 - -0.2521183).abs() < 1e-4);
    assert!((embeddings[1][765] as f64 - 0.4667896).abs() < 1e-4);
    assert!((embeddings[1][766] as f64 - 0.14219822).abs() < 1e-4);
    assert!((embeddings[1][767] as f64 - 0.3986863).abs() < 1e-4);

    Ok(())
}

#[test]
fn sbert_t5() -> anyhow::Result<()> {
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::SentenceT5Base)
        .create_model()?;

    let sentences = ["This is an example sentence", "Each sentence is converted"];
    let embeddings = model.encode(&sentences)?;

    assert!((embeddings[0][0] as f64 - -0.00904849).abs() < 1e-4);
    assert!((embeddings[0][1] as f64 - 0.0191336).abs() < 1e-4);
    assert!((embeddings[0][2] as f64 - 0.02657794).abs() < 1e-4);
    assert!((embeddings[0][765] as f64 - -0.00876413).abs() < 1e-4);
    assert!((embeddings[0][766] as f64 - -0.05602207).abs() < 1e-4);
    assert!((embeddings[0][767] as f64 - -0.02163094).abs() < 1e-4);

    assert!((embeddings[1][0] as f64 - -0.00785422).abs() < 1e-4);
    assert!((embeddings[1][1] as f64 - 0.03018173).abs() < 1e-4);
    assert!((embeddings[1][2] as f64 - 0.03129675).abs() < 1e-4);
    assert!((embeddings[1][765] as f64 - -0.01246878).abs() < 1e-4);
    assert!((embeddings[1][766] as f64 - -0.06240674).abs() < 1e-4);
    assert!((embeddings[1][767] as f64 - -0.00590969).abs() < 1e-4);

    Ok(())
}

#[test]
fn keyword_extraction_cosine_similarity() -> anyhow::Result<()> {
    let keyword_extraction_config = KeywordExtractionConfig {
        sentence_embeddings_config: SentenceEmbeddingsConfig::from(
            SentenceEmbeddingsModelType::AllMiniLmL6V2,
        ),
        scorer_type: KeywordScorerType::CosineSimilarity,
        ngram_range: (1, 1),
        num_keywords: 5,
        ..Default::default()
    };

    let keyword_extraction_model = KeywordExtractionModel::new(keyword_extraction_config)?;

    let input = [
        "Rust is a multi-paradigm, general-purpose programming language. \
 Rust emphasizes performance, type safety, and concurrency. Rust enforces memory safety—that is, \
 that all references point to valid memory—without requiring the use of a garbage collector or \
 reference counting present in other memory-safe languages. To simultaneously enforce \
 memory safety and prevent concurrent data races, Rust's borrow checker tracks the object lifetime \
 and variable scope of all references in a program during compilation. Rust is popular for \
 systems programming but also offers high-level features including functional programming constructs.",
        "Machine learning (ML) is a field of inquiry devoted to understanding and building methods \
 that 'learn', that is, methods that leverage data to improve performance on some set of tasks.\
 It is seen as a part of artificial intelligence. Machine learning algorithms build a model \
 based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so."
    ];
    // Credits: Wikimedia foundation https://en.wikipedia.org/wiki/Rust_(programming_language)
    // Credits: Wikimedia foundation https://en.wikipedia.org/wiki/Machine_learning

    let keywords = keyword_extraction_model.predict(&input)?;

    assert_eq!(keywords.len(), 2);
    assert_eq!(keywords[0].len(), 5);
    assert_eq!(keywords[0][0].text, "rust");
    assert!((keywords[0][0].score - 0.5091).abs() < 1e-4);
    assert_eq!(keywords[0][1].text, "programming");
    assert!((keywords[0][1].score - 0.3573).abs() < 1e-4);
    assert_eq!(keywords[0][2].text, "concurrency");
    assert!((keywords[0][2].score - 0.3382).abs() < 1e-4);
    assert_eq!(keywords[1].len(), 5);
    assert_eq!(keywords[1][0].text, "ml");
    assert!((keywords[1][0].score - 0.4100).abs() < 1e-4);
    assert_eq!(keywords[1][1].text, "learning");
    assert!((keywords[1][1].score - 0.3855).abs() < 1e-4);
    assert_eq!(keywords[1][2].text, "machine");
    assert!((keywords[1][2].score - 0.3633).abs() < 1e-4);

    Ok(())
}

#[test]
fn keyword_extraction_maximal_margin_relevance() -> anyhow::Result<()> {
    let keyword_extraction_config = KeywordExtractionConfig {
        sentence_embeddings_config: SentenceEmbeddingsConfig::from(
            SentenceEmbeddingsModelType::AllMiniLmL6V2,
        ),
        scorer_type: KeywordScorerType::MaximalMarginRelevance,
        ngram_range: (1, 1),
        num_keywords: 5,
        ..Default::default()
    };

    let keyword_extraction_model = KeywordExtractionModel::new(keyword_extraction_config)?;

    let input = [
        "Rust is a multi-paradigm, general-purpose programming language. \
 Rust emphasizes performance, type safety, and concurrency. Rust enforces memory safety—that is, \
 that all references point to valid memory—without requiring the use of a garbage collector or \
 reference counting present in other memory-safe languages. To simultaneously enforce \
 memory safety and prevent concurrent data races, Rust's borrow checker tracks the object lifetime \
 and variable scope of all references in a program during compilation. Rust is popular for \
 systems programming but also offers high-level features including functional programming constructs.",
        "Machine learning (ML) is a field of inquiry devoted to understanding and building methods \
 that 'learn', that is, methods that leverage data to improve performance on some set of tasks.\
 It is seen as a part of artificial intelligence. Machine learning algorithms build a model \
 based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so."
    ];

    let keywords = keyword_extraction_model.predict(&input)?;

    assert_eq!(keywords.len(), 2);
    assert_eq!(keywords[0].len(), 5);
    assert_eq!(keywords[0][0].text, "rust");
    assert!((keywords[0][0].score - 0.5091).abs() < 1e-4);
    assert_eq!(keywords[0][1].text, "programming");
    assert!((keywords[0][1].score - 0.3573).abs() < 1e-4);
    assert_eq!(keywords[0][2].text, "concurrency");
    assert!((keywords[0][2].score - 0.3382).abs() < 1e-4);
    assert_eq!(keywords[1].len(), 5);
    assert_eq!(keywords[1][0].text, "ml");
    assert!((keywords[1][0].score - 0.4100).abs() < 1e-4);
    assert_eq!(keywords[1][1].text, "machine");
    assert!((keywords[1][1].score - 0.3633).abs() < 1e-4);
    assert_eq!(keywords[1][2].text, "algorithms");
    assert!((keywords[1][2].score - 0.3519).abs() < 1e-4);

    Ok(())
}

#[test]
fn keyword_extraction_max_sum() -> anyhow::Result<()> {
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

    let input = [
        "Rust is a multi-paradigm, general-purpose programming language. \
 Rust emphasizes performance, type safety, and concurrency. Rust enforces memory safety—that is, \
 that all references point to valid memory—without requiring the use of a garbage collector or \
 reference counting present in other memory-safe languages. To simultaneously enforce \
 memory safety and prevent concurrent data races, Rust's borrow checker tracks the object lifetime \
 and variable scope of all references in a program during compilation. Rust is popular for \
 systems programming but also offers high-level features including functional programming constructs.",
        "Machine learning (ML) is a field of inquiry devoted to understanding and building methods \
 that 'learn', that is, methods that leverage data to improve performance on some set of tasks.\
 It is seen as a part of artificial intelligence. Machine learning algorithms build a model \
 based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so."
    ];

    let keywords = keyword_extraction_model.predict(&input)?;

    assert_eq!(keywords.len(), 2);
    assert_eq!(keywords[0].len(), 5);
    assert_eq!(keywords[0][0].text, "rust");
    assert!((keywords[0][0].score - 0.5091).abs() < 1e-4);
    assert_eq!(keywords[0][1].text, "concurrency");
    assert!((keywords[0][1].score - 0.3382).abs() < 1e-4);
    assert_eq!(keywords[0][2].text, "languages");
    assert!((keywords[0][2].score - 0.2851).abs() < 1e-4);
    assert_eq!(keywords[1].len(), 5);
    assert_eq!(keywords[1][0].text, "ml");
    assert!((keywords[1][0].score - 0.4100).abs() < 1e-4);
    assert_eq!(keywords[1][1].text, "algorithms");
    assert!((keywords[1][1].score - 0.3519).abs() < 1e-4);
    assert_eq!(keywords[1][2].text, "intelligence");
    assert!((keywords[1][2].score - 0.2492).abs() < 1e-4);

    Ok(())
}

#[test]
fn keyword_extraction_cosine_similarity_n_grams() -> anyhow::Result<()> {
    let keyword_extraction_config = KeywordExtractionConfig {
        sentence_embeddings_config: SentenceEmbeddingsConfig::from(
            SentenceEmbeddingsModelType::AllMiniLmL6V2,
        ),
        scorer_type: KeywordScorerType::CosineSimilarity,
        ngram_range: (1, 2),
        num_keywords: 5,
        ..Default::default()
    };

    let keyword_extraction_model = KeywordExtractionModel::new(keyword_extraction_config)?;

    let input = [
        "Rust is a multi-paradigm, general-purpose programming language. \
 Rust emphasizes performance, type safety, and concurrency. Rust enforces memory safety—that is, \
 that all references point to valid memory—without requiring the use of a garbage collector or \
 reference counting present in other memory-safe languages. To simultaneously enforce \
 memory safety and prevent concurrent data races, Rust's borrow checker tracks the object lifetime \
 and variable scope of all references in a program during compilation. Rust is popular for \
 systems programming but also offers high-level features including functional programming constructs.",
        "Machine learning (ML) is a field of inquiry devoted to understanding and building methods \
 that 'learn', that is, methods that leverage data to improve performance on some set of tasks.\
 It is seen as a part of artificial intelligence. Machine learning algorithms build a model \
 based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so."
    ];

    let keywords = keyword_extraction_model.predict(&input)?;

    assert_eq!(keywords.len(), 2);
    assert_eq!(keywords[0].len(), 5);
    assert_eq!(keywords[0][0].text, "rust enforces");
    assert!((keywords[0][0].score - 0.6471).abs() < 1e-4);
    assert_eq!(keywords[0][1].text, "rust");
    assert!((keywords[0][1].score - 0.5091).abs() < 1e-4);
    assert_eq!(keywords[0][2].text, "programming language");
    assert!((keywords[0][2].score - 0.4868).abs() < 1e-4);
    assert_eq!(keywords[1].len(), 5);
    assert_eq!(keywords[1][0].text, "machine learning");
    assert!((keywords[1][0].score - 0.5683).abs() < 1e-4);
    assert_eq!(keywords[1][1].text, "learning algorithms");
    assert!((keywords[1][1].score - 0.4827).abs() < 1e-4);
    assert_eq!(keywords[1][2].text, "artificial intelligence");
    assert!((keywords[1][2].score - 0.4633).abs() < 1e-4);

    Ok(())
}
