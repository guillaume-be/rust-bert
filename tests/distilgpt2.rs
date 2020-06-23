use rust_bert::gpt2::{
    GPT2LMHeadModel, Gpt2Config, Gpt2ConfigResources, Gpt2MergesResources, Gpt2ModelResources,
    Gpt2VocabResources,
};
use rust_bert::pipelines::generation::{Cache, LMHeadModel};
use rust_bert::resources::{download_resource, RemoteResource, Resource};
use rust_bert::Config;
use rust_tokenizers::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use tch::{nn, Device, Tensor};

#[test]
fn distilgpt2_lm_model() -> failure::Fallible<()> {
    //    Resources paths
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        Gpt2ConfigResources::DISTIL_GPT2,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        Gpt2VocabResources::DISTIL_GPT2,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        Gpt2MergesResources::DISTIL_GPT2,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        Gpt2ModelResources::DISTIL_GPT2,
    ));
    let config_path = download_resource(&config_resource)?;
    let vocab_path = download_resource(&vocab_resource)?;
    let merges_path = download_resource(&merges_resource)?;
    let weights_path = download_resource(&weights_resource)?;

    //    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_file(
        vocab_path.to_str().unwrap(),
        merges_path.to_str().unwrap(),
        false,
    );
    let config = Gpt2Config::from_file(config_path);
    let gpt2_model = GPT2LMHeadModel::new(&vs.root(), &config);
    vs.load(weights_path)?;

    //    Define input
    let input = ["One two three four five six seven eight nine ten eleven"];
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
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    //    Forward pass
    let (output, _, past, _, _) = gpt2_model
        .forward_t(
            &Some(input_tensor),
            Cache::None,
            &None,
            &None,
            &None,
            &None,
            None,
            &None,
            false,
        )
        .unwrap();

    let next_word_id = output.get(0).get(-1).argmax(-1, true).int64_value(&[0]);
    let next_word = tokenizer.decode(vec![next_word_id], true, true);

    assert_eq!(output.size(), vec!(1, 11, 50257));
    match past {
        Cache::GPT2Cache(past) => {
            assert!(past.is_some());
            assert_eq!(past.as_ref().unwrap().len(), config.n_layer as usize);
            assert_eq!(
                past.as_ref().unwrap()[0].size(),
                vec!(2, 1, config.n_head, 11, 64)
            );
        }
        _ => panic!("Wrong cache returned for GPT2"),
    }
    assert!(
        (output.double_value(&[0, output.size()[1] - 1, next_word_id]) - (-48.7065)).abs() < 1e-4
    );
    assert_eq!(next_word_id, 14104i64);
    assert_eq!(next_word, String::from(" twelve"));

    Ok(())
}
