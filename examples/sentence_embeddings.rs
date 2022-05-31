extern crate anyhow;

use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;

fn main() -> anyhow::Result<()> {
    // Set-up sentence embeddings model
    let model = SentenceEmbeddingsModel::new(Default::default())?;

    // Define input
    let sentences = ["this is an example sentence", "each sentence is converted"];

    // Generate Embeddings
    let embeddings = model.encode(&sentences)?;
    println!("{:?}", embeddings);
    Ok(())
}
