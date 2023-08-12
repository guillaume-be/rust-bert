#[cfg(feature = "hf-tokenizers")]
mod tests {
    use rust_bert::gpt2::{Gpt2ConfigResources, Gpt2ModelResources};
    use rust_bert::pipelines::common::{ModelResource, ModelType, TokenizerOption};
    use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
    use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
    use rust_bert::resources::{LocalResource, RemoteResource, ResourceProvider};
    use std::fs::File;
    use std::io::Write;
    use tch::Device;
    use tempfile::TempDir;

    #[test]
    fn gpt2_generation() -> anyhow::Result<()> {
        let model_resource = Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));
        let config_resource = Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
        let dummy_vocab_resource = Box::new(LocalResource {
            local_path: Default::default(),
        });
        let tokenizer_resource = Box::new(RemoteResource::from_pretrained((
            "gpt2/tokenizer",
            "https://huggingface.co/gpt2/resolve/main/tokenizer.json",
        )));

        let generate_config = TextGenerationConfig {
            model_type: ModelType::GPT2,
            model_resource: ModelResource::Torch(model_resource),
            config_resource,
            vocab_resource: dummy_vocab_resource,
            merges_resource: None,
            max_length: Some(20),
            do_sample: false,
            num_beams: 5,
            temperature: 1.2,
            device: Device::Cpu,
            num_return_sequences: 3,
            ..Default::default()
        };

        // Create tokenizer
        let tmp_dir = TempDir::new()?;
        let special_token_map_path = tmp_dir.path().join("special_token_map.json");
        let mut tmp_file = File::create(&special_token_map_path)?;
        writeln!(
            tmp_file,
            r#"{{"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>"}}"#
        )?;

        let tokenizer_path = tokenizer_resource.get_local_path()?;
        let tokenizer =
            TokenizerOption::from_hf_tokenizer_file(tokenizer_path, special_token_map_path)?;

        let model = TextGenerationModel::new_with_tokenizer(generate_config, tokenizer)?;

        let input_context = "The dog";
        let output = model.generate(&[input_context], None);

        assert_eq!(output.len(), 3);
        assert_eq!(
            output[0],
            "The dog was found in the backyard of a home in the 6200 block of South Main Street."
        );
        assert_eq!(
            output[1],
            "The dog was found in the backyard of a home in the 6500 block of South Main Street."
        );
        assert_eq!(
            output[2],
            "The dog was found in the backyard of a home in the 6200 block of South Main Street,"
        );
        Ok(())
    }

    #[test]
    fn distilbert_question_answering() -> anyhow::Result<()> {
        // Create tokenizer
        let tmp_dir = TempDir::new()?;
        let special_token_map_path = tmp_dir.path().join("special_token_map.json");
        let mut tmp_file = File::create(&special_token_map_path)?;

        writeln!(
            tmp_file,
            r#"{{"pad_token": "[PAD]", "sep_token": "[SEP]", "cls_token": "[CLS]", "mask_token": "[MASK]", "unk_token": "[UNK]"}}"#
        )?;
        let tokenizer_resource = Box::new(RemoteResource::from_pretrained((
            "distilbert-base-cased-distilled-squad/tokenizer",
            "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/tokenizer.json",
        )));
        let tokenizer_path = tokenizer_resource.get_local_path()?;
        let tokenizer =
            TokenizerOption::from_hf_tokenizer_file(tokenizer_path, special_token_map_path)?;

        //    Set-up question answering model
        let qa_model = QuestionAnsweringModel::new_with_tokenizer(Default::default(), tokenizer)?;

        //    Define input
        let question = String::from("Where does Amy live ?");
        let context = String::from("Amy lives in Amsterdam");
        let qa_input = QaInput { question, context };

        let answers = qa_model.predict(&[qa_input], 1, 32);

        assert_eq!(answers.len(), 1usize);
        assert_eq!(answers[0].len(), 1usize);
        assert_eq!(answers[0][0].start, 13);
        assert_eq!(answers[0][0].end, 22);
        assert!((answers[0][0].score - 0.9978).abs() < 1e-4);
        assert_eq!(answers[0][0].answer, "Amsterdam");

        Ok(())
    }
}
