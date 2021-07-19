use rust_bert::mbart::{
    MBartConfig, MBartConfigResources, MBartGenerator, MBartModel, MBartModelResources,
    MBartVocabResources,
};
use rust_bert::pipelines::generation_utils::{GenerateConfig, LanguageGenerator};
use rust_bert::resources::{RemoteResource, Resource};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{MBart50Tokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, Device, Tensor};

#[test]
fn mbart_lm_model() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        MBartConfigResources::MBART50_MANY_TO_MANY,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        MBartVocabResources::MBART50_MANY_TO_MANY,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        MBartModelResources::MBART50_MANY_TO_MANY,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer = MBart50Tokenizer::from_file(vocab_path.to_str().unwrap(), false)?;
    let config = MBartConfig::from_file(config_path);
    let mbart_model = MBartModel::new(&vs.root() / "model", &config);
    vs.load(weights_path)?;

    //    Define input
    let input = ["One two three four"];
    let tokenized_input = tokenizer.encode_list(&input, 128, &TruncationStrategy::LongestFirst, 0);
    let max_len = tokenized_input
        .iter()
        .map(|input| input.token_ids.len())
        .max()
        .unwrap();
    let tokenized_input = tokenized_input
        .iter()
        .map(|input| input.token_ids.clone())
        .map(|mut input| {
            input.extend(vec![0; max_len - input.len()]);
            input
        })
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let model_output =
        mbart_model.forward_t(Some(&input_tensor), None, None, None, None, None, false);
    assert_eq!(model_output.decoder_output.size(), vec!(1, 5, 1024));
    assert_eq!(
        model_output.encoder_hidden_state.unwrap().size(),
        vec!(1, 5, 1024)
    );
    assert!((model_output.decoder_output.double_value(&[0, 0, 0]) - -0.8936).abs() < 1e-4);
    Ok(())
}

#[test]
fn mbart_translation() -> anyhow::Result<()> {
    //    Resources paths
    let generate_config = GenerateConfig {
        max_length: 56,
        model_resource: Resource::Remote(RemoteResource::from_pretrained(
            MBartModelResources::MBART50_MANY_TO_MANY,
        )),
        config_resource: Resource::Remote(RemoteResource::from_pretrained(
            MBartConfigResources::MBART50_MANY_TO_MANY,
        )),
        vocab_resource: Resource::Remote(RemoteResource::from_pretrained(
            MBartVocabResources::MBART50_MANY_TO_MANY,
        )),
        merges_resource: Resource::Remote(RemoteResource::from_pretrained(
            MBartVocabResources::MBART50_MANY_TO_MANY,
        )),
        do_sample: false,
        num_beams: 3,
        ..Default::default()
    };
    let model = MBartGenerator::new(generate_config)?;

    let input_context = ">>en<< The quick brown fox jumps over the lazy dog.";
    let target_language = model.get_tokenizer().convert_tokens_to_ids([">>de<<"])[0];

    let output = model.generate(
        Some(&[input_context]),
        None,
        None,
        None,
        None,
        target_language,
        None,
        false,
    );

    assert_eq!(output.len(), 1);
    assert_eq!(
        output[0].text,
        " Der schnelle braune Fuchs springt über den faulen Hund."
    );

    Ok(())
}
