use std::path::PathBuf;
use rust_bert::common::resources::{CACHE_DIRECTORY, download_file};

#[test]
fn download_cached_config() -> failure::Fallible<()> {
    let config_path = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json";
//    let config_path = "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-cased-distilled-squad-rust_model.ot";
    let mut target: PathBuf = CACHE_DIRECTORY.clone().to_path_buf();
    target.push("config.json");
//    target.push("model.ot");
    let _ = download_file(config_path, &target);
    Ok(())
}