use rust_bert::sbert::{
    SBertModel, UsingAlbert, UsingBert, UsingDistilBert, UsingRoberta, UsingT5,
};

/// Download model:
///   ```sh
///   git lfs install
///   git -C resources clone https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased
///   ```
/// Prepare model:
///   ```sh
///   python ./utils/convert_model.py resources/distiluse-base-multilingual-cased/pytorch_model.bin --prefix distilbert.
///   python ./utils/convert_model.py resources/distiluse-base-multilingual-cased/2_Dense/pytorch_model.bin --suffix
///   ```
/// Test results based on:
///   https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased
#[test]
#[ignore]
fn sbert_distilbert() -> anyhow::Result<()> {
    let device = tch::Device::cuda_if_available();
    let sbert =
        SBertModel::<UsingDistilBert>::new("resources/distiluse-base-multilingual-cased", device)?;

    let sentences = ["This is an example sentence", "Each sentence is converted"];
    let embeddings = sbert.encode(&sentences)?;

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

/// Download model:
///   ```sh
///   git lfs install
///   git -C resources clone https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5
///   ```
/// Prepare model:
///   ```sh
///   python ./utils/convert_model.py resources/msmarco-bert-base-dot-v5/pytorch_model.bin
///   ```
/// Test results based on:
///   https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5
#[test]
#[ignore]
fn sbert_bert() -> anyhow::Result<()> {
    let device = tch::Device::cuda_if_available();
    let sbert = SBertModel::<UsingBert>::new("resources/msmarco-bert-base-dot-v5", device)?;

    // Note that sentences are lowercased here as it seems required in sbert example
    // even though sentence_bert_config.json has `"do_lower_case":false`
    let sentences = ["this is an example sentence", "each sentence is converted"];
    let embeddings = sbert.encode(&sentences)?;

    assert!((embeddings[0][0] as f64 - -0.153149).abs() < 1e-4);
    assert!((embeddings[0][1] as f64 - 0.12944254).abs() < 1e-4);
    assert!((embeddings[0][2] as f64 - -0.03903132).abs() < 1e-4);
    assert!((embeddings[0][765] as f64 - 0.09051804).abs() < 1e-4);
    assert!((embeddings[0][766] as f64 - -0.12790504).abs() < 1e-4);
    assert!((embeddings[0][767] as f64 - -0.02097229).abs() < 1e-4);

    assert!((embeddings[1][0] as f64 - -0.01726065).abs() < 1e-4);
    assert!((embeddings[1][1] as f64 - 0.17196466).abs() < 1e-4);
    assert!((embeddings[1][2] as f64 - 0.14468355).abs() < 1e-4);
    assert!((embeddings[1][765] as f64 - 0.26710105).abs() < 1e-4);
    assert!((embeddings[1][766] as f64 - 0.15404876).abs() < 1e-4);
    assert!((embeddings[1][767] as f64 - -0.14560737).abs() < 1e-4);

    Ok(())
}

/// Download model:
///   ```sh
///   git lfs install
///   git -C resources clone https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
///   ```
/// Prepare model:
///   ```sh
///   python ./utils/convert_model.py resources/all-MiniLM-L12-v2/pytorch_model.bin
///   ```
/// Test results based on:
///   https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2
#[test]
#[ignore]
fn sbert_bert_small() -> anyhow::Result<()> {
    let device = tch::Device::cuda_if_available();
    let sbert = SBertModel::<UsingBert>::new("resources/all-MiniLM-L12-v2", device)?;

    // Note that sentences are lowercased here as it seems required in sbert example
    // even though sentence_bert_config.json has `"do_lower_case":false`
    let sentences = ["this is an example sentence", "each sentence is converted"];
    let embeddings = sbert.encode(&sentences)?;

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

/// Download model:
///   ```sh
///   git lfs install
///   git -C resources clone https://huggingface.co/sentence-transformers/all-distilroberta-v1
///   ```
/// Prepare model:
///   ```sh
///   python ./utils/convert_model.py resources/all-distilroberta-v1/pytorch_model.bin
///   ```
/// Test results based on:
///   https://huggingface.co/sentence-transformers/all-distilroberta-v1
#[test]
#[ignore]
fn sbert_distilroberta() -> anyhow::Result<()> {
    let device = tch::Device::cuda_if_available();
    let sbert = SBertModel::<UsingRoberta>::new("resources/all-distilroberta-v1", device)?;

    let sentences = ["This is an example sentence", "Each sentence is converted"];
    let embeddings = sbert.encode(&sentences)?;

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

/// Download model:
///   ```sh
///   git lfs install
///   git -C resources clone https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2
///   ```
/// Prepare model:
///   ```sh
///   python ./utils/convert_model.py resources/paraphrase-albert-small-v2/pytorch_model.bin
///   ```
/// Test results based on:
///   https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2
#[test]
#[ignore]
fn sbert_albert() -> anyhow::Result<()> {
    let device = tch::Device::cuda_if_available();
    let sbert = SBertModel::<UsingAlbert>::new("resources/paraphrase-albert-small-v2", device)?;

    // Note that sentences are lowercased here as it seems required in sbert example
    // even though sentence_bert_config.json has `"do_lower_case":false`
    let sentences = ["this is an example sentence", "each sentence is converted"];
    let embeddings = sbert.encode(&sentences)?;

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

/// Download model:
///   ```sh
///   git lfs install
///   git -C resources clone https://huggingface.co/sentence-transformers/sentence-t5-base
///   ```
/// Prepare model:
///   ```sh
///   python ./utils/convert_model.py resources/sentence-t5-base/pytorch_model.bin
///   python ./utils/convert_model.py resources/sentence-t5-base/2_Dense/pytorch_model.bin --suffix
///   ```
/// Test results based on:
///   https://huggingface.co/sentence-transformers/sentence-t5-base
#[test]
#[ignore]
fn sbert_t5() -> anyhow::Result<()> {
    let device = tch::Device::cuda_if_available();
    let sbert = SBertModel::<UsingT5>::new("resources/sentence-t5-base", device)?;

    let sentences = ["This is an example sentence", "Each sentence is converted"];
    let embeddings = sbert.encode(&sentences)?;

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
