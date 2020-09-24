use rust_bert::pipelines::generation::{GenerateConfig, LanguageGenerator, XLNetGenerator};
use rust_bert::resources::{RemoteResource, Resource};
use rust_bert::xlnet::{
    XLNetConfig, XLNetConfigResources, XLNetForSequenceClassification, XLNetLMHeadModel,
    XLNetModelResources, XLNetVocabResources,
};
use rust_bert::Config;
use rust_tokenizers::{Tokenizer, TruncationStrategy, Vocab, XLNetTokenizer};
use std::collections::HashMap;
use tch::{nn, no_grad, Device, Kind, Tensor};

#[test]
fn xlnet_lm_model() -> anyhow::Result<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        XLNetConfigResources::XLNET_BASE_CASED,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        XLNetVocabResources::XLNET_BASE_CASED,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        XLNetModelResources::XLNET_BASE_CASED,
    ));
    let config_path = config_resource.get_local_path()?;
    let vocab_path = vocab_resource.get_local_path()?;
    let weights_path = weights_resource.get_local_path()?;

    //    Set-up masked LM model
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    let tokenizer: XLNetTokenizer =
        XLNetTokenizer::from_file(vocab_path.to_str().unwrap(), false, true)?;
    let config = XLNetConfig::from_file(config_path);
    let xlnet_model = XLNetLMHeadModel::new(&vs.root(), &config);
    vs.load(weights_path)?;

    //    Define input
    let input = ["One two three four"];
    let tokenized_input =
        tokenizer.encode_list(input.to_vec(), 128, &TruncationStrategy::LongestFirst, 0);
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
        .map(|input| Tensor::of_slice(&(input[..input.len() - 2])))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    // Forward pass
    let perm_mask = Tensor::zeros(&[1, 4, 4], (Kind::Float, device));
    let _ = perm_mask.narrow(2, 3, 1).fill_(1.0);

    let target_mapping = Tensor::zeros(&[1, 1, 4], (Kind::Float, device));
    let _ = target_mapping.narrow(2, 3, 1).fill_(1.0);
    let model_output = no_grad(|| {
        xlnet_model
            .forward_t(
                Some(&input_tensor),
                None,
                None,
                Some(perm_mask.as_ref()),
                Some(target_mapping.as_ref()),
                None,
                None,
                false,
            )
            .unwrap()
    });

    let index_1 = model_output.lm_logits.get(0).argmax(1, false);
    let word_1 = tokenizer.vocab().id_to_token(&index_1.int64_value(&[]));

    println!("{}", word_1);
    assert_eq!(model_output.lm_logits.size(), vec!(1, 1, 32000));
    assert!((model_output.lm_logits.double_value(&[0, 0, 139]) - -5.3240).abs() < 1e-4);
    Ok(())
}

#[test]
fn xlnet_generation_beam_search() -> anyhow::Result<()> {
    //    Set-up masked LM model
    //    Set-up masked LM model
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        XLNetConfigResources::XLNET_BASE_CASED,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        XLNetVocabResources::XLNET_BASE_CASED,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        XLNetVocabResources::XLNET_BASE_CASED,
    ));
    let model_resource = Resource::Remote(RemoteResource::from_pretrained(
        XLNetModelResources::XLNET_BASE_CASED,
    ));

    let generate_config = GenerateConfig {
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        max_length: 32,
        do_sample: false,
        num_beams: 3,
        temperature: 1.0,
        num_return_sequences: 1,
        ..Default::default()
    };
    let model = XLNetGenerator::new(generate_config)?;

    let input_context = "Once upon a time,";
    let output = model.generate(Some(vec![input_context]), None);

    assert_eq!(output.len(), 1);
    assert_eq!(
        output[0],
        " Once upon a time, there was a time when there was only one man in the world who could do all the things he wanted to do. There was no one who"
    );

    Ok(())
}
