// use crate::marian::marian_model::MarianModelPreset;
// use crate::marian::{
//     MarianConfigResources, MarianModelResources, MarianSourceLanguages, MarianSpmResources,
//     MarianTargetLanguages, MarianVocabResources,
// };
// use crate::pipelines::common::ModelType;
// use crate::pipelines::translation::{Language, TranslationModelConfig};
// use crate::resources::{RemoteResource, Resource};
// use std::borrow::Cow;
//
// impl MarianModelPreset {
//     pub const ENGLISH2GERMAN: TranslationModelConfig<[Language; 1], [Language; 1]> =
//         TranslationModelConfig {
//             model_type: ModelType::Marian,
//             model_resource: Resource::Remote(RemoteResource {
//                 url: Cow::Borrowed(MarianModelResources::ENGLISH2GERMAN.0),
//                 cache_subdir: Cow::Borrowed(MarianModelResources::ENGLISH2GERMAN.1),
//             }),
//             config_resource: Resource::Remote(RemoteResource {
//                 url: Cow::Borrowed(MarianConfigResources::ENGLISH2GERMAN.0),
//                 cache_subdir: Cow::Borrowed(MarianConfigResources::ENGLISH2GERMAN.1),
//             }),
//             vocab_resource: Resource::Remote(RemoteResource {
//                 url: Cow::Borrowed(MarianVocabResources::ENGLISH2GERMAN.0),
//                 cache_subdir: Cow::Borrowed(MarianVocabResources::ENGLISH2GERMAN.1),
//             }),
//             merges_resource: Resource::Remote(RemoteResource {
//                 url: Cow::Borrowed(MarianSpmResources::ENGLISH2GERMAN.0),
//                 cache_subdir: Cow::Borrowed(MarianSpmResources::ENGLISH2GERMAN.1),
//             }),
//             source_languages: MarianSourceLanguages::ENGLISH2GERMAN,
//             target_languages: MarianTargetLanguages::ENGLISH2GERMAN,
//         };
// }
