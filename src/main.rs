use crate::distilbert::DistilBertConfig;
use std::path::Path;
use std::env;

mod distilbert;


fn main() {

    let config_path = env::var("distilbert_config_path").unwrap();
    let config_path = Path::new(&config_path);

    let config = DistilBertConfig::from_file(config_path);
    let cuda_available = tch::Cuda::cudnn_is_available();
    println!("{:?}", cuda_available);

    println!("{:?}", config);

}
