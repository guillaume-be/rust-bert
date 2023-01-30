use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};

fn main() -> anyhow::Result<()> {
    // Set-up sentence embeddings model
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()?;

    // Define input
    let sentences = ["this is an example sentence", "each sentence is converted"];

    // Generate Embeddings
    let embeddings = model.encode(&sentences)?;
    println!("{embeddings:?}");
    Ok(())
}
