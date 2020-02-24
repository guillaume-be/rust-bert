extern crate failure;
extern crate dirs;

use std::path::PathBuf;
use rust_bert::pipelines::ner::NERModel;
use tch::Device;


fn main() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("bert-ner");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let weights_path = &home.as_path().join("model.ot");

//    Set-up model
    let device = Device::cuda_if_available();
    let ner_model = NERModel::new(vocab_path,
                                  config_path,
                                  weights_path, device)?;

//    Define input
    let input = [
        "My name is Amy. I live in Paris.",
        "Paris is a city in France."
    ];

//    Run model
    let output = ner_model.predict(input.to_vec());
    for entity in output {
        println!("{:?}", entity);
    }

    Ok(())
}