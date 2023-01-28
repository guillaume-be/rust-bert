use crate::pipelines::common::ModelType;
use crate::pipelines::translation::Language;
use std::fmt::Debug;
use tch::Device;

#[cfg(feature = "remote")]
use crate::{
    pipelines::translation::{TranslationConfig, TranslationModel},
    resources::ResourceProvider,
    RustBertError,
};

#[derive(Clone, Copy, PartialEq)]
enum ModelSize {
    Medium,
    Large,
    XLarge,
}

/// # Translation Model Builder
/// Allows the user to provide a variable set of inputs and identifies the best translation model that fulfills the constraints provided
/// by the user. Options that can provided by the user include:
/// - Target device (CPU/CUDA)
/// - Model size (medium, large or extra large)
/// - source languages to support (as an array of [`Language`])
/// - target languages to support (as an array of [`Language`])
/// - model type ([`ModelType`], supported models include `Marian`, `T5`, `MBart50` or `M2M100`)
///
/// The logic for selecting the most appropriate model is as follows:
/// - If not specified, the model will be executed on a CUDA device if available, otherwise on the CPU
/// - If the model type is specified (e.g. `Marian`), a model with this architecture will be created. The compatibility of the model
/// with the source and target languages will be verified, and the builder will error if the settings provided are not supported.
/// - If the model size is specified, a model of the corresponding size class (computational budget) will be created. The compatibility of the model
/// with the source and target languages will be verified, and the builder will error if the settings provided are not supported.
/// - If no source or target languages are provided, a multilingual M2M100 model will be returned
/// - If no model type is provided, an average sized-model (Marian) will be returned if a pretrained model exists that covers the requested source/target languages provided.
/// Otherwise a M2M100 multi-lingual model will be returned.
///
/// The options for the builder are provided with dedicated "builder function", the call to `create_model()` creates a model
/// from the builder.
///
/// # Example
///
/// ```no_run
/// use rust_bert::pipelines::translation::{Language, TranslationModelBuilder};
/// fn main() -> anyhow::Result<()> {
///     let model = TranslationModelBuilder::new()
///         .with_source_languages(vec![Language::English])
///         .with_target_languages(vec![Language::Spanish, Language::French, Language::Italian])
///         .create_model()?;
///
///     let input_context_1 = "This is a sentence to be translated";
///     let input_context_2 = "The dog did not wake up.";
///
///     let output =
///         model.translate(&[input_context_1, input_context_2], None, Language::Spanish)?;
///
///     for sentence in output {
///         println!("{}", sentence);
///     }
///     Ok(())
/// }
/// ```
pub struct TranslationModelBuilder {
    model_type: Option<ModelType>,
    source_languages: Option<Vec<Language>>,
    target_languages: Option<Vec<Language>>,
    device: Option<Device>,
    model_size: Option<ModelSize>,
}

impl Default for TranslationModelBuilder {
    fn default() -> Self {
        TranslationModelBuilder::new()
    }
}

impl TranslationModelBuilder {
    /// Build a new `TranslationModelBuilder`
    ///
    /// # Returns
    /// * `TranslationModelBuilder` Translation model builder
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::translation::TranslationModelBuilder;
    /// fn main() -> anyhow::Result<()> {
    ///     let model = TranslationModelBuilder::new().create_model();
    ///     Ok(())
    /// }
    /// ```
    pub fn new() -> TranslationModelBuilder {
        TranslationModelBuilder {
            model_type: None,
            source_languages: None,
            target_languages: None,
            device: None,
            model_size: None,
        }
    }

    /// Specify the device for the translation model
    ///
    /// # Arguments
    /// * `device` - [`tch::Device`] target device for the model.
    ///
    /// # Returns
    /// * `TranslationModelBuilder` Translation model builder
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::translation::TranslationModelBuilder;
    /// use tch::Device;
    /// fn main() -> anyhow::Result<()> {
    ///     let model = TranslationModelBuilder::new()
    ///         .with_device(Device::Cuda(0))
    ///         .create_model();
    ///     Ok(())
    /// }
    /// ```
    pub fn with_device(&mut self, device: Device) -> &mut Self {
        self.device = Some(device);
        self
    }

    /// Specify the model type for the translation model
    ///
    /// # Arguments
    /// * `model_type` - [`ModelType`] type of translation model to load.
    ///
    /// # Returns
    /// * `TranslationModelBuilder` Translation model builder
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::common::ModelType;
    /// use rust_bert::pipelines::translation::TranslationModelBuilder;
    /// fn main() -> anyhow::Result<()> {
    ///     let model = TranslationModelBuilder::new()
    ///         .with_model_type(ModelType::M2M100)
    ///         .create_model();
    ///     Ok(())
    /// }
    /// ```
    pub fn with_model_type(&mut self, model_type: ModelType) -> &mut Self {
        self.model_type = Some(model_type);
        self
    }

    /// Use a medium-sized translation model (Marian-based)
    ///
    /// # Returns
    /// * `TranslationModelBuilder` Translation model builder
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::translation::TranslationModelBuilder;
    /// fn main() -> anyhow::Result<()> {
    ///     let model = TranslationModelBuilder::new()
    ///         .with_medium_model()
    ///         .create_model();
    ///     Ok(())
    /// }
    /// ```
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

    /// Use a large translation model (M2M100, 418M parameters-based)
    ///
    /// # Returns
    /// * `TranslationModelBuilder` Translation model builder
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::translation::TranslationModelBuilder;
    /// fn main() -> anyhow::Result<()> {
    ///     let model = TranslationModelBuilder::new()
    ///         .with_large_model()
    ///         .create_model();
    ///     Ok(())
    /// }
    /// ```
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

    /// Use a very large translation model (M2M100, 1.2B parameters-based)
    ///
    /// # Returns
    /// * `TranslationModelBuilder` Translation model builder
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::translation::TranslationModelBuilder;
    /// fn main() -> anyhow::Result<()> {
    ///     let model = TranslationModelBuilder::new()
    ///         .with_xlarge_model()
    ///         .create_model();
    ///     Ok(())
    /// }
    /// ```
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

    /// Specify the source languages the model should support
    ///
    /// # Arguments
    /// * `source_languages` - Array-like of [`Language`] the model should be able to translate from
    ///
    /// # Returns
    /// * `TranslationModelBuilder` Translation model builder
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::translation::Language;
    /// use rust_bert::pipelines::translation::TranslationModelBuilder;
    /// fn main() -> anyhow::Result<()> {
    ///     let model = TranslationModelBuilder::new()
    ///         .with_source_languages([Language::French, Language::Italian])
    ///         .create_model();
    ///     Ok(())
    /// }
    /// ```
    pub fn with_source_languages<S>(&mut self, source_languages: S) -> &mut Self
    where
        S: AsRef<[Language]> + Debug,
    {
        self.source_languages = Some(source_languages.as_ref().to_vec());
        self
    }

    /// Specify the target languages the model should support
    ///
    /// # Arguments
    /// * `target_languages` - Array-like of [`Language`] the model should be able to translate to
    ///
    /// # Returns
    /// * `TranslationModelBuilder` Translation model builder
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::translation::Language;
    /// use rust_bert::pipelines::translation::TranslationModelBuilder;
    /// fn main() -> anyhow::Result<()> {
    ///     let model = TranslationModelBuilder::new()
    ///         .with_target_languages([
    ///             Language::Japanese,
    ///             Language::Korean,
    ///             Language::ChineseMandarin,
    ///         ])
    ///         .create_model();
    ///     Ok(())
    /// }
    /// ```
    pub fn with_target_languages<T>(&mut self, target_languages: T) -> &mut Self
    where
        T: AsRef<[Language]> + Debug,
    {
        self.target_languages = Some(target_languages.as_ref().to_vec());
        self
    }

    /// Creates the translation model based on the specifications provided
    ///
    /// # Returns
    /// * `TranslationModel` Generated translation model
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::pipelines::translation::Language;
    /// use rust_bert::pipelines::translation::TranslationModelBuilder;
    /// fn main() -> anyhow::Result<()> {
    ///     let model = TranslationModelBuilder::new()
    ///         .with_target_languages([
    ///             Language::Japanese,
    ///             Language::Korean,
    ///             Language::ChineseMandarin,
    ///         ])
    ///         .create_model();
    ///     Ok(())
    /// }
    /// ```
    #[cfg(feature = "remote")]
    pub fn create_model(&self) -> Result<TranslationModel, RustBertError> {
        let device = self.device.unwrap_or_else(Device::cuda_if_available);

        let translation_resources = match (
            &self.model_type,
            &self.source_languages,
            &self.target_languages,
        ) {
            (Some(ModelType::M2M100), source_languages, target_languages) => {
                match self.model_size {
                    Some(value) if value == ModelSize::XLarge => {
                        model_fetchers::get_m2m100_xlarge_resources(
                            source_languages.as_ref(),
                            target_languages.as_ref(),
                        )?
                    }
                    _ => model_fetchers::get_m2m100_large_resources(
                        source_languages.as_ref(),
                        target_languages.as_ref(),
                    )?,
                }
            }
            (Some(ModelType::MBart), source_languages, target_languages) => {
                model_fetchers::get_mbart50_resources(
                    source_languages.as_ref(),
                    target_languages.as_ref(),
                )?
            }
            (Some(ModelType::Marian), source_languages, target_languages) => {
                model_fetchers::get_marian_model(
                    source_languages.as_ref(),
                    target_languages.as_ref(),
                )?
            }
            (None, source_languages, target_languages) => model_fetchers::get_default_model(
                &self.model_size,
                source_languages.as_ref(),
                target_languages.as_ref(),
            )?,
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
            Some(translation_resources.merges_resource),
            translation_resources.source_languages,
            translation_resources.target_languages,
            device,
        );
        TranslationModel::new(translation_config)
    }
}

#[cfg(feature = "remote")]
mod model_fetchers {
    use super::*;
    use crate::{
        m2m_100::{
            M2M100ConfigResources, M2M100MergesResources, M2M100ModelResources,
            M2M100SourceLanguages, M2M100TargetLanguages, M2M100VocabResources,
        },
        marian::{
            MarianConfigResources, MarianModelResources, MarianSourceLanguages, MarianSpmResources,
            MarianTargetLanguages, MarianVocabResources,
        },
        mbart::{
            MBartConfigResources, MBartModelResources, MBartSourceLanguages, MBartTargetLanguages,
            MBartVocabResources,
        },
        resources::RemoteResource,
    };

    pub(super) struct TranslationResources<R>
    where
        R: ResourceProvider + Send + 'static,
    {
        pub(super) model_type: ModelType,
        pub(super) model_resource: R,
        pub(super) config_resource: R,
        pub(super) vocab_resource: R,
        pub(super) merges_resource: R,
        pub(super) source_languages: Vec<Language>,
        pub(super) target_languages: Vec<Language>,
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

    pub(super) fn get_default_model(
        model_size: &Option<ModelSize>,
        source_languages: Option<&Vec<Language>>,
        target_languages: Option<&Vec<Language>>,
    ) -> Result<TranslationResources<RemoteResource>, RustBertError> {
        Ok(match get_marian_model(source_languages, target_languages) {
            Ok(marian_resources) => marian_resources,
            Err(_) => match model_size {
                Some(value) if value == &ModelSize::XLarge => {
                    get_m2m100_xlarge_resources(source_languages, target_languages)?
                }
                _ => get_m2m100_large_resources(source_languages, target_languages)?,
            },
        })
    }

    pub(super) fn get_marian_model(
        source_languages: Option<&Vec<Language>>,
        target_languages: Option<&Vec<Language>>,
    ) -> Result<TranslationResources<RemoteResource>, RustBertError> {
        let (resources, source_languages, target_languages) =
            if let (Some(source_languages), Some(target_languages)) =
                (source_languages, target_languages)
            {
                match (source_languages.as_slice(), target_languages.as_slice()) {
                    ([Language::English], [Language::German]) => {
                        get_marian_resources!(ENGLISH2GERMAN)
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
                    ([Language::German], [Language::French]) => {
                        get_marian_resources!(GERMAN2FRENCH)
                    }
                    ([Language::French], [Language::German]) => {
                        get_marian_resources!(FRENCH2GERMAN)
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
                return Err(RustBertError::InvalidConfigurationError(
                    "Source and target languages must be provided for Marian models".to_string(),
                ));
            };

        Ok(TranslationResources {
            model_type: ModelType::Marian,
            model_resource: RemoteResource::from_pretrained(resources.0),
            config_resource: RemoteResource::from_pretrained(resources.1),
            vocab_resource: RemoteResource::from_pretrained(resources.2),
            merges_resource: RemoteResource::from_pretrained(resources.3),
            source_languages,
            target_languages,
        })
    }

    pub(super) fn get_mbart50_resources(
        source_languages: Option<&Vec<Language>>,
        target_languages: Option<&Vec<Language>>,
    ) -> Result<TranslationResources<RemoteResource>, RustBertError> {
        if let Some(source_languages) = source_languages {
            if !source_languages
                .iter()
                .all(|lang| MBartSourceLanguages::MBART50_MANY_TO_MANY.contains(lang))
            {
                return Err(RustBertError::ValueError(format!(
                    "{:?} not in list of supported languages: {:?}",
                    source_languages,
                    MBartSourceLanguages::MBART50_MANY_TO_MANY
                )));
            }
        }

        if let Some(target_languages) = target_languages {
            if !target_languages
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
            model_resource: RemoteResource::from_pretrained(
                MBartModelResources::MBART50_MANY_TO_MANY,
            ),
            config_resource: RemoteResource::from_pretrained(
                MBartConfigResources::MBART50_MANY_TO_MANY,
            ),
            vocab_resource: RemoteResource::from_pretrained(
                MBartVocabResources::MBART50_MANY_TO_MANY,
            ),
            merges_resource: RemoteResource::from_pretrained(
                MBartVocabResources::MBART50_MANY_TO_MANY,
            ),
            source_languages: MBartSourceLanguages::MBART50_MANY_TO_MANY.to_vec(),
            target_languages: MBartTargetLanguages::MBART50_MANY_TO_MANY.to_vec(),
        })
    }

    pub(super) fn get_m2m100_large_resources(
        source_languages: Option<&Vec<Language>>,
        target_languages: Option<&Vec<Language>>,
    ) -> Result<TranslationResources<RemoteResource>, RustBertError> {
        if let Some(source_languages) = source_languages {
            if !source_languages
                .iter()
                .all(|lang| M2M100SourceLanguages::M2M100_418M.contains(lang))
            {
                return Err(RustBertError::ValueError(format!(
                    "{:?} not in list of supported languages: {:?}",
                    source_languages,
                    M2M100SourceLanguages::M2M100_418M
                )));
            }
        }

        if let Some(target_languages) = target_languages {
            if !target_languages
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
            model_resource: RemoteResource::from_pretrained(M2M100ModelResources::M2M100_418M),
            config_resource: RemoteResource::from_pretrained(M2M100ConfigResources::M2M100_418M),
            vocab_resource: RemoteResource::from_pretrained(M2M100VocabResources::M2M100_418M),
            merges_resource: RemoteResource::from_pretrained(M2M100MergesResources::M2M100_418M),
            source_languages: M2M100SourceLanguages::M2M100_418M.to_vec(),
            target_languages: M2M100TargetLanguages::M2M100_418M.to_vec(),
        })
    }

    pub(super) fn get_m2m100_xlarge_resources(
        source_languages: Option<&Vec<Language>>,
        target_languages: Option<&Vec<Language>>,
    ) -> Result<TranslationResources<RemoteResource>, RustBertError> {
        if let Some(source_languages) = source_languages {
            if !source_languages
                .iter()
                .all(|lang| M2M100SourceLanguages::M2M100_1_2B.contains(lang))
            {
                return Err(RustBertError::ValueError(format!(
                    "{:?} not in list of supported languages: {:?}",
                    source_languages,
                    M2M100SourceLanguages::M2M100_1_2B
                )));
            }
        }

        if let Some(target_languages) = target_languages {
            if !target_languages
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
            model_resource: RemoteResource::from_pretrained(M2M100ModelResources::M2M100_1_2B),
            config_resource: RemoteResource::from_pretrained(M2M100ConfigResources::M2M100_1_2B),
            vocab_resource: RemoteResource::from_pretrained(M2M100VocabResources::M2M100_1_2B),
            merges_resource: RemoteResource::from_pretrained(M2M100MergesResources::M2M100_1_2B),
            source_languages: M2M100SourceLanguages::M2M100_1_2B.to_vec(),
            target_languages: M2M100TargetLanguages::M2M100_1_2B.to_vec(),
        })
    }
}
