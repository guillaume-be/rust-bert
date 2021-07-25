use rust_bert::m2m_100::{
    M2M100Config, M2M100ConfigResources, M2M100Generator, M2M100MergesResources, M2M100Model,
    M2M100ModelResources, M2M100VocabResources,
};
use rust_bert::pipelines::generation_utils::{GenerateConfig, LanguageGenerator};
use rust_bert::resources::{RemoteResource, Resource};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{M2M100Tokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, Device, Tensor};

#[test]
fn m2m100_lm_model() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        M2M100ConfigResources::M2M100_418M,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        M2M100VocabResources::M2M100_418M,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        M2M100MergesResources::M2M100_418M,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        M2M100ModelResources::M2M100_418M,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let merges_path = merges_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer = M2M100Tokenizer::from_files(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        false,
    )?;
    let config = M2M100Config::from_file(config_path);
    let m2m100_model = M2M100Model::new(&vs.root() / "model", &config);
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
        m2m100_model.forward_t(Some(&input_tensor), None, None, None, None, None, false);
    assert_eq!(model_output.decoder_output.size(), vec!(1, 5, 1024));
    assert_eq!(
        model_output.encoder_hidden_state.unwrap().size(),
        vec!(1, 5, 1024)
    );
    assert!(
        (model_output.decoder_output.double_value(&[0, 0, 0]) - -2.047429323196411).abs() < 1e-4
    );
    Ok(())
}

#[test]
fn m2m100_translation() -> anyhow::Result<()> {
    //    Resources paths
    let generate_config = GenerateConfig {
        max_length: 56,
        model_resource: Resource::Remote(RemoteResource::from_pretrained(
            M2M100ModelResources::M2M100_418M,
        )),
        config_resource: Resource::Remote(RemoteResource::from_pretrained(
            M2M100ConfigResources::M2M100_418M,
        )),
        vocab_resource: Resource::Remote(RemoteResource::from_pretrained(
            M2M100VocabResources::M2M100_418M,
        )),
        merges_resource: Resource::Remote(RemoteResource::from_pretrained(
            M2M100MergesResources::M2M100_418M,
        )),
        do_sample: false,
        num_beams: 3,
        ..Default::default()
    };
    let model = M2M100Generator::new(generate_config)?;

    let input_context = ">>en.<< The dog did not wake up.";
    let target_language = model.get_tokenizer().convert_tokens_to_ids([">>es.<<"])[0];

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
    assert_eq!(output[0].text, " El perro no se despertó.");

    Ok(())
}
