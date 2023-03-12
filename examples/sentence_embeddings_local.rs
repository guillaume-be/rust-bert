use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsBuilder;

/// Download model:
///   ```sh
///   git lfs install
///   git -C resources clone https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
///   ```
/// Prepare model:
///   ```sh
///   python ./utils/convert_model.py resources/all-MiniLM-L12-v2/pytorch_model.bin
///   ```
///
/// For models missing the prefix in their saved weights (e.g. Distil-based models), the
/// conversion needs to be updated to include this prefix so that the weights can be found:
///   ```sh
///   python ./utils/convert_model.py resources/path/to/pytorch_model.bin --prefix distilbert.
///   ```
///
/// For models including a dense projection layer (e.g. Distil-based models), these weights
/// need to be converted as well:
/// ```sh
///   python ../utils/convert_model.py  resources/path/to/2_Dense/pytorch_model.bin --suffix
/// ```
fn main() -> anyhow::Result<()> {
    // Set-up sentence embeddings model
    let model = SentenceEmbeddingsBuilder::local("resources/all-MiniLM-L12-v2")
        .with_device(tch::Device::cuda_if_available())
        .create_model()?;

    // Define input
    let sentences = ["this is an example sentence", "each sentence is converted"];

    // Generate Embeddings
    let embeddings = model.encode(&sentences)?;
    println!("{embeddings:?}");
    Ok(())
}
