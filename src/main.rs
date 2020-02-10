use std::path::Path;
use std::env;
use tch::Device;
use crate::distilbert::sentiment::SentimentClassifier;

#[macro_use]
extern crate failure;

mod distilbert;

fn main() -> failure::Fallible<()> {

//    Resources paths
    let config_path = env::var("distilbert_config_path").unwrap();
    let vocab_path = env::var("distilbert_vocab_path").unwrap();
    let weights_path = env::var("distilbert_weights_path").unwrap();

    let config_path = Path::new(&config_path);
    let vocab_path = Path::new(&vocab_path);
    let weights_path = Path::new(&weights_path);

//    Set-up classifier
    let device = Device::Cpu;
    let sentiment_classifier = SentimentClassifier::new(vocab_path,
                                                        config_path,
                                                        weights_path, device)?;

//    Get sentiments
    let input = [
        "This was a great movie",
        "This movie was not great",
        "Very mixed feeling about this, but all in all not bad",
    ];

    let output = sentiment_classifier.predict(input.to_vec());
    println!("{:?}", output);

    Ok(())
}
