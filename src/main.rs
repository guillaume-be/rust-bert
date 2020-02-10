use std::path::Path;
use std::env;
use tch::Device;
use rust_bert::distilbert::sentiment::SentimentClassifier;
extern crate failure;

fn main() -> failure::Fallible<()> {

//    Resources paths
    let config_path = env::var("distilbert_config_path").unwrap();
    let vocab_path = env::var("distilbert_vocab_path").unwrap();
    let weights_path = env::var("distilbert_weights_path").unwrap();

    let config_path = Path::new(&config_path);
    let vocab_path = Path::new(&vocab_path);
    let weights_path = Path::new(&weights_path);

//    Set-up classifier
    let device = Device::cuda_if_available();
    let sentiment_classifier = SentimentClassifier::new(vocab_path,
                                                        config_path,
                                                        weights_path, device)?;

//    Get sentiments
    let input = [
        "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
        "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
        "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
    ];

    let output = sentiment_classifier.predict(input.to_vec());
    println!("{:?}", output);

    Ok(())
}
