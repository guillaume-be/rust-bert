use std::path::PathBuf;
use rust_bert::common::resources::{CACHE_DIRECTORY, download_dependency, Dependency, RemoteDependency};
use rust_bert::distilbert::DistilBertModelDependencies;

#[test]
fn test_download_dependency() -> failure::Fallible<()> {
//    Given
    let config_path = "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json";
    let mut target: PathBuf = CACHE_DIRECTORY.to_path_buf();
    target.push("config.json");

//    When
    let remote_dependency = Dependency::Remote(RemoteDependency::new(config_path, target));

//    Then
    let _ = download_dependency(remote_dependency);
    Ok(())
}

#[test]
fn test_download_dependency_distilbert() -> failure::Fallible<()> {
//    Given
    let model_dependency = Dependency::Remote(RemoteDependency::from_pretrained(DistilBertModelDependencies::DISTIL_BERT_SST2));
    println!("{:?}", model_dependency.get_local_path());

//    Then
    let _ = download_dependency(model_dependency);
    Ok(())
}