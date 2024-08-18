#[cfg(feature = "onnx")]
mod tests {
    extern crate anyhow;

    use rust_bert::m2m_100::{M2M100SourceLanguages, M2M100TargetLanguages};
    use rust_bert::pipelines::common::{ModelResource, ModelType, ONNXModelResources};
    use rust_bert::pipelines::masked_language::{MaskedLanguageConfig, MaskedLanguageModel};
    use rust_bert::pipelines::ner::NERModel;
    use rust_bert::pipelines::question_answering::{
        QaInput, QuestionAnsweringConfig, QuestionAnsweringModel,
    };
    use rust_bert::pipelines::sentiment::{SentimentModel, SentimentPolarity};
    use rust_bert::pipelines::sequence_classification::SequenceClassificationConfig;
    use rust_bert::pipelines::text_generation::{TextGenerationConfig, TextGenerationModel};
    use rust_bert::pipelines::token_classification::{
        LabelAggregationOption, TokenClassificationConfig,
    };
    use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
    use rust_bert::resources::RemoteResource;
    use tch::Device;

    #[test]
    fn onnx_masked_lm() -> anyhow::Result<()> {
        let masked_lm = MaskedLanguageModel::new(MaskedLanguageConfig::new(
            ModelType::Bert,
            ModelResource::ONNX(ONNXModelResources {
                encoder_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/bert-base-uncased-for-masked-lm/resolve/main/model.onnx",
                    "onnx-bert-base-uncased-for-masked-lm",
                ))),
                ..Default::default()
            }),
            RemoteResource::new(
                "https://huggingface.co/optimum/bert-base-uncased-for-masked-lm/resolve/main/config.json",
                "onnx-bert-base-uncased-for-masked-lm",
            ),
            RemoteResource::new(
                "https://huggingface.co/optimum/bert-base-uncased-for-masked-lm/resolve/main/vocab.txt",
                "onnx-bert-base-uncased-for-masked-lm",
            ),
            None,
            false,
            None,
            None,
            Some(String::from("<mask>")),
        ))?;
        let input = [
            "Hello I am a <mask> student",
            "Paris is the <mask> of France. It is <mask> in Europe.",
        ];
        let output = masked_lm.predict(input)?;

        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 1);
        assert_eq!(output[0][0].text, "university");
        assert_eq!(output[0][0].id, 2755);
        assert!((output[0][0].score - 10.0135).abs() < 1e-4);
        assert_eq!(output[1].len(), 2);
        assert_eq!(output[1][0].text, "capital");
        assert_eq!(output[1][0].id, 2364);
        assert!((output[1][0].score - 19.4008).abs() < 1e-4);
        assert_eq!(output[1][1].text, "located");
        assert_eq!(output[1][1].id, 1388);
        assert!((output[1][1].score - 10.8547).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn onnx_question_answering() -> anyhow::Result<()> {
        let qa_model = QuestionAnsweringModel::new(QuestionAnsweringConfig::new(
            ModelType::Roberta,
            ModelResource::ONNX(ONNXModelResources {
                encoder_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/roberta-base-squad2/resolve/main/model.onnx",
                    "onnx-roberta-base-squad2",
                ))),
                ..Default::default()
            }),
            RemoteResource::new(
                "https://huggingface.co/optimum/roberta-base-squad2/resolve/main/config.json",
                "onnx-roberta-base-squad2",
            ),
            RemoteResource::new(
                "https://huggingface.co/optimum/roberta-base-squad2/resolve/main/vocab.json",
                "onnx-roberta-base-squad2",
            ),
            Some(RemoteResource::new(
                "https://huggingface.co/optimum/roberta-base-squad2/resolve/main/merges.txt",
                "onnx-roberta-base-squad2",
            )),
            false,
            None,
            None,
        ))?;
        let question = String::from("Where does Amy live ?");
        let context = String::from("Amy lives in Amsterdam");
        let qa_input = QaInput { question, context };

        let output = qa_model.predict(&[qa_input], 1, 32);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 1);
        assert_eq!(output[0][0].answer, " Amsterdam");
        assert!((output[0][0].score - 0.9898).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn onnx_sequence_classification() -> anyhow::Result<()> {
        let classification_model = SentimentModel::new(SequenceClassificationConfig::new(
            ModelType::DistilBert,
            ModelResource::ONNX(ONNXModelResources {
                encoder_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/model.onnx",
                    "onnx-distilbert-base-uncased-finetuned-sst-2-english",
                ))),
                ..Default::default()
            }),
            RemoteResource::new(
                "https://huggingface.co/optimum/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json",
                "onnx-distilbert-base-uncased-finetuned-sst-2-english",
            ),
            RemoteResource::new(
                "https://huggingface.co/optimum/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/vocab.txt",
                "onnx-distilbert-base-uncased-finetuned-sst-2-english",
            ),
            None,
            true,
            None,
            None,
        ))?;
        let input = [
            "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
            "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
            "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
        ];
        let output = classification_model.predict(input);
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].polarity, SentimentPolarity::Positive);
        assert!((output[0].score - 0.9981).abs() < 1e-4);
        assert_eq!(output[1].polarity, SentimentPolarity::Negative);
        assert!((output[1].score - 0.9927).abs() < 1e-4);
        assert_eq!(output[2].polarity, SentimentPolarity::Positive);
        assert!((output[2].score - 0.9997).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn onnx_token_classification() -> anyhow::Result<()> {
        let token_classification_model = NERModel::new(TokenClassificationConfig::new(
            ModelType::Bert,
            ModelResource::ONNX(ONNXModelResources {
                encoder_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/bert-base-NER/resolve/main/model.onnx",
                    "onnx-bert-base-NER",
                ))),
                ..Default::default()
            }),
            RemoteResource::new(
                "https://huggingface.co/optimum/bert-base-NER/resolve/main/config.json",
                "onnx-bert-base-NER",
            ),
            RemoteResource::new(
                "https://huggingface.co/optimum/bert-base-NER/resolve/main/vocab.txt",
                "onnx-bert-base-NER",
            ),
            None,
            false,
            None,
            None,
            LabelAggregationOption::First,
        ))?;
        let input = ["Asked John Smith about Acme Corp", "Let's go to New York!"];
        let output = token_classification_model.predict_full_entities(&input);
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 2);
        assert_eq!(output[0][0].word, "John Smith");
        assert_eq!(output[0][0].label, "PER");
        assert!((output[0][0].score - 0.9992).abs() < 1e-4);
        assert_eq!(output[0][1].word, "Acme Corp");
        assert_eq!(output[0][1].label, "ORG");
        assert!((output[0][1].score - 0.0001).abs() < 1e-4);
        assert_eq!(output[1].len(), 1);
        assert_eq!(output[1][0].word, "New York");
        assert_eq!(output[1][0].label, "LOC");
        assert!((output[1][0].score - 0.9987).abs() < 1e-4);

        Ok(())
    }

    #[test]
    fn onnx_text_generation() -> anyhow::Result<()> {
        let text_generation_model = TextGenerationModel::new(TextGenerationConfig {
            model_type: ModelType::GPT2,
            model_resource: ModelResource::ONNX(ONNXModelResources {
                encoder_resource: None,
                decoder_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/gpt2/resolve/main/decoder_model.onnx",
                    "onnx-gpt2",
                ))),
                decoder_with_past_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/gpt2/resolve/main/decoder_with_past_model.onnx",
                    "onnx-gpt2",
                ))),
            }),
            config_resource: Box::new(RemoteResource::new(
                "https://huggingface.co/optimum/gpt2/resolve/main/config.json",
                "onnx-gpt2",
            )),
            vocab_resource: Box::new(RemoteResource::new(
                "https://huggingface.co/gpt2/resolve/main/vocab.json",
                "onnx-gpt2",
            )),
            merges_resource: Some(Box::new(RemoteResource::new(
                "https://huggingface.co/gpt2/resolve/main/merges.txt",
                "onnx-gpt2",
            ))),
            max_length: Some(30),
            do_sample: false,
            num_beams: 1,
            temperature: 1.0,
            num_return_sequences: 1,
            ..Default::default()
        })?;
        let prompts = ["It was a very nice and sunny"];
        let output = text_generation_model.generate(&prompts, None)?;
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], "It was a very nice and sunny day. I was very happy with the weather. I was very happy with the weather. I was very happy with");
        Ok(())
    }

    #[test]
    fn onnx_translation() -> anyhow::Result<()> {
        let translation_model = TranslationModel::new(TranslationConfig::new(
            ModelType::M2M100,
            ModelResource::ONNX(ONNXModelResources {
                encoder_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/m2m100_418M/resolve/e775f50e63b178d82b8d736fc43fcf5ef15d2f6c/encoder_model.onnx",
                    "onnx-m2m100_418M",
                ))),
                decoder_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/m2m100_418M/resolve/e775f50e63b178d82b8d736fc43fcf5ef15d2f6c/decoder_model.onnx",
                    "onnx-m2m100_418M",
                ))),
                decoder_with_past_resource: Some(Box::new(RemoteResource::new(
                    "https://huggingface.co/optimum/m2m100_418M/resolve/e775f50e63b178d82b8d736fc43fcf5ef15d2f6c/decoder_with_past_model.onnx",
                    "onnx-m2m100_418M",
                ))),
            }),
            RemoteResource::new(
                "https://huggingface.co/optimum/m2m100_418M/resolve/main/config.json",
                "onnx-m2m100_418M",
            ),
            RemoteResource::new(
                "https://huggingface.co/optimum/m2m100_418M/resolve/main/vocab.json",
                "onnx-m2m100_418M",
            ),
            Some(RemoteResource::new(
                "https://huggingface.co/optimum/m2m100_418M/resolve/main/sentencepiece.bpe.model",
                "onnx-m2m100_418M",
            )),
            M2M100SourceLanguages::M2M100_418M,
            M2M100TargetLanguages::M2M100_418M,
            Device::cuda_if_available(),
        ))?;

        let source_sentence = "This sentence will be translated in multiple languages.";

        let mut outputs = Vec::new();
        outputs.extend(translation_model.translate(
            &[source_sentence],
            Language::English,
            Language::French,
        )?);
        outputs.extend(translation_model.translate(
            &[source_sentence],
            Language::English,
            Language::Spanish,
        )?);
        outputs.extend(translation_model.translate(
            &[source_sentence],
            Language::English,
            Language::Hindi,
        )?);

        assert_eq!(outputs.len(), 3);
        assert_eq!(
            outputs[0],
            " Cette phrase sera traduite en plusieurs langues."
        );
        assert_eq!(outputs[1], " Esta frase se traducirá en varios idiomas.");
        assert_eq!(outputs[2], " यह वाक्यांश कई भाषाओं में अनुवादित किया जाएगा।");

        Ok(())
    }
}
