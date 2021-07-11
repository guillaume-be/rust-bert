use crate::m2m_100::{
    M2M100ConfigResources, M2M100MergesResources, M2M100ModelResources, M2M100SourceLanguages,
    M2M100TargetLanguages, M2M100VocabResources,
};
use crate::marian::{
    MarianConfigResources, MarianModelResources, MarianSourceLanguages, MarianSpmResources,
    MarianTargetLanguages, MarianVocabResources,
};
use crate::mbart::{
    MBartConfigResources, MBartModelResources, MBartSourceLanguages, MBartTargetLanguages,
    MBartVocabResources,
};
use crate::pipelines::common::ModelType;
use crate::pipelines::translation::{Language, TranslationConfig, TranslationModel};
use crate::resources::{RemoteResource, Resource};
use crate::RustBertError;
use std::fmt::Debug;
use tch::Device;

struct TranslationResources {
    model_type: ModelType,
    model_resource: Resource,
    config_resource: Resource,
    vocab_resource: Resource,
    merges_resource: Resource,
    source_languages: Vec<Language>,
    target_languages: Vec<Language>,
}

#[derive(Clone, Copy, PartialEq)]
enum ModelSize {
    Medium,
    Large,
    XLarge,
}

pub struct TranslationModelBuilder<S, T>
where
    S: AsRef<[Language]> + Debug,
    T: AsRef<[Language]> + Debug,
{
    model_type: Option<ModelType>,
    source_languages: Option<S>,
    target_languages: Option<T>,
    device: Option<Device>,
    model_size: Option<ModelSize>,
}

macro_rules! get_marian_resources {
    ($name:ident) => {
        (
            (
                MarianModelResources::$name,
                MarianConfigResources::$name,
                MarianVocabResources::$name,
                MarianSpmResources::$name,
            ),
            MarianSourceLanguages::$name.iter().cloned().collect(),
            MarianTargetLanguages::$name.iter().cloned().collect(),
        )
    };
}

impl<S, T> TranslationModelBuilder<S, T>
where
    S: AsRef<[Language]> + Debug,
    T: AsRef<[Language]> + Debug,
{
    pub fn new() -> TranslationModelBuilder<S, T> {
        TranslationModelBuilder {
            model_type: None,
            source_languages: None,
            target_languages: None,
            device: None,
            model_size: None,
        }
    }

    pub fn with_device(&mut self, device: Device) -> &mut Self {
        self.device = Some(device);
        self
    }

    pub fn with_model_type(&mut self, model_type: ModelType) -> &mut Self {
        self.model_type = Some(model_type);
        self
    }

    pub fn with_medium_model(&mut self) -> &mut Self {
        if let Some(model_type) = self.model_type {
            if model_type != ModelType::Marian {
                eprintln!(
                    "Model selection overwritten: was {:?}, replaced by {:?} (medium model selected)",
                    self.model_type.unwrap(),
                    ModelType::Marian
                );
            }
        }
        self.model_type = Some(ModelType::Marian);
        self.model_size = Some(ModelSize::Medium);
        self
    }

    pub fn with_large_model(&mut self) -> &mut Self {
        if let Some(model_type) = self.model_type {
            if model_type != ModelType::M2M100 {
                eprintln!(
                    "Model selection overwritten: was {:?}, replaced by {:?} (large model selected)",
                    self.model_type.unwrap(),
                    ModelType::M2M100
                );
            }
        }
        self.model_type = Some(ModelType::M2M100);
        self.model_size = Some(ModelSize::Large);
        self
    }

    pub fn with_xlarge_model(&mut self) -> &mut Self {
        if let Some(model_type) = self.model_type {
            if model_type != ModelType::M2M100 {
                eprintln!(
                    "Model selection overwritten: was {:?}, replaced by {:?} (xlarge model selected)",
                    self.model_type.unwrap(),
                    ModelType::M2M100
                );
            }
        }
        self.model_type = Some(ModelType::M2M100);
        self.model_size = Some(ModelSize::XLarge);
        self
    }

    pub fn with_source_languages(&mut self, source_languages: S) -> &mut Self {
        self.source_languages = Some(source_languages);
        self
    }

    pub fn with_target_languages(&mut self, target_languages: T) -> &mut Self {
        self.target_languages = Some(target_languages);
        self
    }

    fn get_default_model(
        &self,
        source_languages: Option<&S>,
        target_languages: Option<&T>,
    ) -> Result<TranslationResources, RustBertError> {
        Ok(
            match self.get_marian_model(source_languages, target_languages) {
                Ok(marian_resources) => marian_resources,
                Err(_) => match self.model_size {
                    Some(value) if value == ModelSize::XLarge => {
                        self.get_m2m100_xlarge_resources(source_languages, target_languages)?
                    }
                    _ => self.get_m2m100_large_resources(source_languages, target_languages)?,
                },
            },
        )
    }

    fn get_marian_model(
        &self,
        source_languages: Option<&S>,
        target_languages: Option<&T>,
    ) -> Result<TranslationResources, RustBertError> {
        let (resources, source_languages, target_languages) =
            if let (Some(source_languages), Some(target_languages)) =
                (source_languages, target_languages)
            {
                match (source_languages.as_ref(), target_languages.as_ref()) {
                    ([Language::English], [Language::German]) => {
                        get_marian_resources!(ENGLISH2RUSSIAN)
                    }
                    ([Language::English], [Language::Russian]) => {
                        get_marian_resources!(ENGLISH2RUSSIAN)
                    }
                    ([Language::English], [Language::Dutch]) => {
                        get_marian_resources!(ENGLISH2DUTCH)
                    }
                    ([Language::English], [Language::ChineseMandarin]) => {
                        get_marian_resources!(ENGLISH2CHINESE)
                    }
                    ([Language::English], [Language::Swedish]) => {
                        get_marian_resources!(ENGLISH2SWEDISH)
                    }
                    ([Language::English], [Language::Arabic]) => {
                        get_marian_resources!(ENGLISH2ARABIC)
                    }
                    ([Language::English], [Language::Hindi]) => {
                        get_marian_resources!(ENGLISH2HINDI)
                    }
                    ([Language::English], [Language::Hebrew]) => {
                        get_marian_resources!(ENGLISH2HEBREW)
                    }
                    ([Language::German], [Language::English]) => {
                        get_marian_resources!(GERMAN2ENGLISH)
                    }
                    ([Language::Russian], [Language::English]) => {
                        get_marian_resources!(RUSSIAN2ENGLISH)
                    }
                    ([Language::Dutch], [Language::English]) => {
                        get_marian_resources!(DUTCH2ENGLISH)
                    }
                    ([Language::ChineseMandarin], [Language::English]) => {
                        get_marian_resources!(CHINESE2ENGLISH)
                    }
                    ([Language::Swedish], [Language::English]) => {
                        get_marian_resources!(SWEDISH2ENGLISH)
                    }
                    ([Language::Arabic], [Language::English]) => {
                        get_marian_resources!(ARABIC2ENGLISH)
                    }
                    ([Language::Hindi], [Language::English]) => {
                        get_marian_resources!(HINDI2ENGLISH)
                    }
                    ([Language::Hebrew], [Language::English]) => {
                        get_marian_resources!(HEBREW2ENGLISH)
                    }
                    ([Language::English], languages)
                        if languages
                            .iter()
                            .all(|lang| MarianTargetLanguages::ENGLISH2ROMANCE.contains(lang)) =>
                    {
                        get_marian_resources!(ENGLISH2ROMANCE)
                    }
                    (languages, [Language::English])
                        if languages
                            .iter()
                            .all(|lang| MarianSourceLanguages::ROMANCE2ENGLISH.contains(lang)) =>
                    {
                        get_marian_resources!(ROMANCE2ENGLISH)
                    }
                    (_, _) => {
                        return Err(RustBertError::InvalidConfigurationError(format!(
                            "No Pretrained Marian configuration found for {:?} to {:?} translation",
                            source_languages, target_languages
                        )));
                    }
                }
            } else {
                return Err(RustBertError::InvalidConfigurationError(format!(
                    "Source and target languages must be provided for Marian models"
                )));
            };

        Ok(TranslationResources {
            model_type: ModelType::Marian,
            model_resource: Resource::Remote(RemoteResource::from_pretrained(resources.0)),
            config_resource: Resource::Remote(RemoteResource::from_pretrained(resources.1)),
            vocab_resource: Resource::Remote(RemoteResource::from_pretrained(resources.2)),
            merges_resource: Resource::Remote(RemoteResource::from_pretrained(resources.3)),
            source_languages,
            target_languages,
        })
    }

    fn get_mbart50_resources(
        &self,
        source_languages: Option<&S>,
        target_languages: Option<&T>,
    ) -> Result<TranslationResources, RustBertError> {
        if let Some(source_languages) = source_languages {
            if !source_languages
                .as_ref()
                .iter()
                .all(|lang| MBartSourceLanguages::MBART50_MANY_TO_MANY.contains(lang))
            {
                return Err(RustBertError::ValueError(format!(
                    "{:?} not in list of supported languages: {:?}",
                    source_languages.as_ref(),
                    MBartSourceLanguages::MBART50_MANY_TO_MANY
                )));
            }
        }

        if let Some(target_languages) = target_languages {
            if !target_languages
                .as_ref()
                .iter()
                .all(|lang| MBartTargetLanguages::MBART50_MANY_TO_MANY.contains(lang))
            {
                return Err(RustBertError::ValueError(format!(
                    "{:?} not in list of supported languages: {:?}",
                    target_languages,
                    MBartTargetLanguages::MBART50_MANY_TO_MANY
                )));
            }
        }

        Ok(TranslationResources {
            model_type: ModelType::MBart,
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
            source_languages: MBartSourceLanguages::MBART50_MANY_TO_MANY
                .iter()
                .cloned()
                .collect(),
            target_languages: MBartTargetLanguages::MBART50_MANY_TO_MANY
                .iter()
                .cloned()
                .collect(),
        })
    }

    fn get_m2m100_large_resources(
        &self,
        source_languages: Option<&S>,
        target_languages: Option<&T>,
    ) -> Result<TranslationResources, RustBertError> {
        if let Some(source_languages) = source_languages {
            if !source_languages
                .as_ref()
                .iter()
                .all(|lang| M2M100SourceLanguages::M2M100_418M.contains(lang))
            {
                return Err(RustBertError::ValueError(format!(
                    "{:?} not in list of supported languages: {:?}",
                    source_languages.as_ref(),
                    M2M100SourceLanguages::M2M100_418M
                )));
            }
        }

        if let Some(target_languages) = target_languages {
            if !target_languages
                .as_ref()
                .iter()
                .all(|lang| M2M100TargetLanguages::M2M100_418M.contains(lang))
            {
                return Err(RustBertError::ValueError(format!(
                    "{:?} not in list of supported languages: {:?}",
                    target_languages,
                    M2M100TargetLanguages::M2M100_418M
                )));
            }
        }

        Ok(TranslationResources {
            model_type: ModelType::M2M100,
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
            source_languages: M2M100SourceLanguages::M2M100_418M.iter().cloned().collect(),
            target_languages: M2M100TargetLanguages::M2M100_418M.iter().cloned().collect(),
        })
    }

    fn get_m2m100_xlarge_resources(
        &self,
        source_languages: Option<&S>,
        target_languages: Option<&T>,
    ) -> Result<TranslationResources, RustBertError> {
        if let Some(source_languages) = source_languages {
            if !source_languages
                .as_ref()
                .iter()
                .all(|lang| M2M100SourceLanguages::M2M100_1_2B.contains(lang))
            {
                return Err(RustBertError::ValueError(format!(
                    "{:?} not in list of supported languages: {:?}",
                    source_languages.as_ref(),
                    M2M100SourceLanguages::M2M100_1_2B
                )));
            }
        }

        if let Some(target_languages) = target_languages {
            if !target_languages
                .as_ref()
                .iter()
                .all(|lang| M2M100TargetLanguages::M2M100_1_2B.contains(lang))
            {
                return Err(RustBertError::ValueError(format!(
                    "{:?} not in list of supported languages: {:?}",
                    target_languages,
                    M2M100TargetLanguages::M2M100_1_2B
                )));
            }
        }

        Ok(TranslationResources {
            model_type: ModelType::M2M100,
            model_resource: Resource::Remote(RemoteResource::from_pretrained(
                M2M100ModelResources::M2M100_1_2B,
            )),
            config_resource: Resource::Remote(RemoteResource::from_pretrained(
                M2M100ConfigResources::M2M100_1_2B,
            )),
            vocab_resource: Resource::Remote(RemoteResource::from_pretrained(
                M2M100VocabResources::M2M100_1_2B,
            )),
            merges_resource: Resource::Remote(RemoteResource::from_pretrained(
                M2M100MergesResources::M2M100_1_2B,
            )),
            source_languages: M2M100SourceLanguages::M2M100_1_2B.iter().cloned().collect(),
            target_languages: M2M100TargetLanguages::M2M100_1_2B.iter().cloned().collect(),
        })
    }

    pub fn create_model(&self) -> Result<TranslationModel, RustBertError> {
        let device = self.device.unwrap_or_else(|| Device::cuda_if_available());

        let translation_resources = match (
            &self.model_type,
            &self.source_languages,
            &self.target_languages,
        ) {
            (Some(ModelType::M2M100), source_languages, target_languages) => {
                match self.model_size {
                    Some(value) if value == ModelSize::XLarge => self.get_m2m100_xlarge_resources(
                        source_languages.as_ref(),
                        target_languages.as_ref(),
                    )?,
                    _ => self.get_m2m100_large_resources(
                        source_languages.as_ref(),
                        target_languages.as_ref(),
                    )?,
                }
            }
            (Some(ModelType::MBart), source_languages, target_languages) => {
                self.get_mbart50_resources(source_languages.as_ref(), target_languages.as_ref())?
            }
            (Some(ModelType::Marian), source_languages, target_languages) => {
                self.get_marian_model(source_languages.as_ref(), target_languages.as_ref())?
            }
            (None, source_languages, target_languages) => {
                self.get_default_model(source_languages.as_ref(), target_languages.as_ref())?
            }
            (_, None, None) | (_, _, None) | (_, None, _) => {
                return Err(RustBertError::InvalidConfigurationError(format!(
                    "Source and target languages must be specified for {:?}",
                    self.model_type.unwrap()
                )));
            }
            (Some(model_type), _, _) => {
                return Err(RustBertError::InvalidConfigurationError(format!(
                    "Automated translation model builder not implemented for {:?}",
                    model_type
                )));
            }
        };

        let translation_config = TranslationConfig::new(
            translation_resources.model_type,
            translation_resources.model_resource,
            translation_resources.config_resource,
            translation_resources.vocab_resource,
            translation_resources.merges_resource,
            translation_resources.source_languages,
            translation_resources.target_languages,
            device,
        );
        TranslationModel::new(translation_config)
    }
}
