// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright 2019-2020 Guillaume Becquin
// Copyright 2020 Maarten van Gompel
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::albert::AlbertForSequenceClassification;
use crate::bart::{
    BartConfigResources, BartForSequenceClassification, BartMergesResources, BartModelResources,
    BartVocabResources,
};
use crate::bert::BertForSequenceClassification;
use crate::distilbert::DistilBertModelClassifier;
use crate::pipelines::common::{ConfigOption, ModelType, TokenizerOption};
use crate::pipelines::sequence_classification::Label;
use crate::resources::{download_resource, RemoteResource, Resource};
use crate::roberta::RobertaForSequenceClassification;
use crate::RustBertError;
use itertools::Itertools;
use rust_tokenizers::{TokenizedInput, TruncationStrategy};
use std::borrow::Borrow;
use tch::nn::VarStore;
use tch::{nn, no_grad, Device, Tensor};

/// # Configuration for ZeroShotClassificationModel
/// Contains information regarding the model to load and device to place the model on.
pub struct ZeroShotClassificationConfig {
    /// Model type
    pub model_type: ModelType,
    /// Model weights resource (default: pretrained BERT model on CoNLL)
    pub model_resource: Resource,
    /// Config resource (default: pretrained BERT model on CoNLL)
    pub config_resource: Resource,
    /// Vocab resource (default: pretrained BERT model on CoNLL)
    pub vocab_resource: Resource,
    /// Merges resource (default: None)
    pub merges_resource: Option<Resource>,
    /// Automatically lower case all input upon tokenization (assumes a lower-cased model)
    pub lower_case: bool,
    /// Flag indicating if the tokenizer should strip accents (normalization). Only used for BERT / ALBERT models
    pub strip_accents: Option<bool>,
    /// Flag indicating if the tokenizer should add a white space before each tokenized input (needed for some Roberta models)
    pub add_prefix_space: Option<bool>,
    /// Device to place the model on (default: CUDA/GPU when available)
    pub device: Device,
}

impl ZeroShotClassificationConfig {
    /// Instantiate a new zero shot classification configuration of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded!)
    /// * model - The `Resource` pointing to the model to load (e.g.  model.ot)
    /// * config - The `Resource' pointing to the model configuration to load (e.g. config.json)
    /// * vocab - The `Resource' pointing to the tokenizer's vocabulary to load (e.g.  vocab.txt/vocab.json)
    /// * vocab - An optional `Resource` tuple (`Option<Resource>`) pointing to the tokenizer's merge file to load (e.g.  merges.txt), needed only for Roberta.
    /// * lower_case - A `bool' indicating whether the tokeniser should lower case all input (in case of a lower-cased model)
    pub fn new(
        model_type: ModelType,
        model_resource: Resource,
        config_resource: Resource,
        vocab_resource: Resource,
        merges_resource: Option<Resource>,
        lower_case: bool,
        strip_accents: impl Into<Option<bool>>,
        add_prefix_space: impl Into<Option<bool>>,
    ) -> ZeroShotClassificationConfig {
        ZeroShotClassificationConfig {
            model_type,
            model_resource,
            config_resource,
            vocab_resource,
            merges_resource,
            lower_case,
            strip_accents: strip_accents.into(),
            add_prefix_space: add_prefix_space.into(),
            device: Device::cuda_if_available(),
        }
    }
}

impl Default for ZeroShotClassificationConfig {
    /// Provides a defaultSST-2 sentiment analysis model (English)
    fn default() -> ZeroShotClassificationConfig {
        ZeroShotClassificationConfig {
            model_type: ModelType::Bart,
            model_resource: Resource::Remote(RemoteResource::from_pretrained(
                BartModelResources::BART_MNLI,
            )),
            config_resource: Resource::Remote(RemoteResource::from_pretrained(
                BartConfigResources::BART_MNLI,
            )),
            vocab_resource: Resource::Remote(RemoteResource::from_pretrained(
                BartVocabResources::BART_MNLI,
            )),
            merges_resource: Some(Resource::Remote(RemoteResource::from_pretrained(
                BartMergesResources::BART_MNLI,
            ))),
            lower_case: false,
            strip_accents: None,
            add_prefix_space: None,
            device: Device::cuda_if_available(),
        }
    }
}

/// # Abstraction that holds one particular zero shot classification model, for any of the supported models
/// The models are using a classification architecture that should be trained on Natural Language Inference.
/// The models should output a Tensor of size > 2 in the label dimension, with the first logit corresponding
/// to contradiction and the last logit corresponding to entailment.
pub enum ZeroShotClassificationOption {
    /// Bart for Sequence Classification
    Bart(BartForSequenceClassification),
    /// Bert for Sequence Classification
    Bert(BertForSequenceClassification),
    /// DistilBert for Sequence Classification
    DistilBert(DistilBertModelClassifier),
    /// Roberta for Sequence Classification
    Roberta(RobertaForSequenceClassification),
    /// XLMRoberta for Sequence Classification
    XLMRoberta(RobertaForSequenceClassification),
    /// Albert for Sequence Classification
    Albert(AlbertForSequenceClassification),
}

impl ZeroShotClassificationOption {
    /// Instantiate a new zero shot classification model of the supplied type.
    ///
    /// # Arguments
    ///
    /// * `model_type` - `ModelType` indicating the model type to load (must match with the actual data to be loaded)
    /// * `p` - `tch::nn::Path` path to the model file to load (e.g. model.ot)
    /// * `config` - A configuration (the model type of the configuration must be compatible with the value for
    /// `model_type`)
    pub fn new<'p, P>(model_type: ModelType, p: P, config: &ConfigOption) -> Self
    where
        P: Borrow<nn::Path<'p>>,
    {
        match model_type {
            ModelType::Bart => {
                if let ConfigOption::Bart(config) = config {
                    ZeroShotClassificationOption::Bart(BartForSequenceClassification::new(
                        p, config,
                    ))
                } else {
                    panic!("You can only supply a BartConfig for Bart!");
                }
            }
            ModelType::Bert => {
                if let ConfigOption::Bert(config) = config {
                    ZeroShotClassificationOption::Bert(BertForSequenceClassification::new(
                        p, config,
                    ))
                } else {
                    panic!("You can only supply a BertConfig for Bert!");
                }
            }
            ModelType::DistilBert => {
                if let ConfigOption::DistilBert(config) = config {
                    ZeroShotClassificationOption::DistilBert(DistilBertModelClassifier::new(
                        p, config,
                    ))
                } else {
                    panic!("You can only supply a DistilBertConfig for DistilBert!");
                }
            }
            ModelType::Roberta => {
                if let ConfigOption::Bert(config) = config {
                    ZeroShotClassificationOption::Roberta(RobertaForSequenceClassification::new(
                        p, config,
                    ))
                } else {
                    panic!("You can only supply a BertConfig for Roberta!");
                }
            }
            ModelType::XLMRoberta => {
                if let ConfigOption::Bert(config) = config {
                    ZeroShotClassificationOption::XLMRoberta(RobertaForSequenceClassification::new(
                        p, config,
                    ))
                } else {
                    panic!("You can only supply a BertConfig for Roberta!");
                }
            }
            ModelType::Albert => {
                if let ConfigOption::Albert(config) = config {
                    ZeroShotClassificationOption::Albert(AlbertForSequenceClassification::new(
                        p, config,
                    ))
                } else {
                    panic!("You can only supply an AlbertConfig for Albert!");
                }
            }
            ModelType::Electra => {
                panic!("SequenceClassification not implemented for Electra!");
            }
            ModelType::Marian => {
                panic!("SequenceClassification not implemented for Marian!");
            }
            ModelType::T5 => {
                panic!("SequenceClassification not implemented for T5!");
            }
        }
    }

    /// Returns the `ModelType` for this SequenceClassificationOption
    pub fn model_type(&self) -> ModelType {
        match *self {
            Self::Bart(_) => ModelType::Bart,
            Self::Bert(_) => ModelType::Bert,
            Self::Roberta(_) => ModelType::Roberta,
            Self::XLMRoberta(_) => ModelType::Roberta,
            Self::DistilBert(_) => ModelType::DistilBert,
            Self::Albert(_) => ModelType::Albert,
        }
    }

    /// Interface method to forward_t() of the particular models.
    pub fn forward_t(
        &self,
        input_ids: Option<Tensor>,
        mask: Option<Tensor>,
        token_type_ids: Option<Tensor>,
        position_ids: Option<Tensor>,
        input_embeds: Option<Tensor>,
        train: bool,
    ) -> Tensor {
        match *self {
            Self::Bart(ref model) => {
                model
                    .forward_t(
                        &input_ids.expect("`input_ids` must be provided for BART models"),
                        mask.as_ref(),
                        None,
                        None,
                        None,
                        train,
                    )
                    .0
            }
            Self::Bert(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        token_type_ids,
                        position_ids,
                        input_embeds,
                        train,
                    )
                    .0
            }
            Self::DistilBert(ref model) => {
                model
                    .forward_t(input_ids, mask, input_embeds, train)
                    .expect("Error in distilbert forward_t")
                    .0
            }
            Self::Roberta(ref model) | Self::XLMRoberta(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        token_type_ids,
                        position_ids,
                        input_embeds,
                        train,
                    )
                    .0
            }
            Self::Albert(ref model) => {
                model
                    .forward_t(
                        input_ids,
                        mask,
                        token_type_ids,
                        position_ids,
                        input_embeds,
                        train,
                    )
                    .0
            }
        }
    }
}

/// # ZeroShotClassificationModel for Zero Shot Classification
pub struct ZeroShotClassificationModel {
    tokenizer: TokenizerOption,
    zero_shot_classifier: ZeroShotClassificationOption,
    var_store: VarStore,
}

impl ZeroShotClassificationModel {
    /// Build a new `ZeroShotClassificationModel`
    ///
    /// # Arguments
    ///
    /// * `config` - `SequenceClassificationConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::sequence_classification::SequenceClassificationModel;
    ///
    /// let model = SequenceClassificationModel::new(Default::default())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        config: ZeroShotClassificationConfig,
    ) -> Result<ZeroShotClassificationModel, RustBertError> {
        let config_path = download_resource(&config.config_resource)?;
        let vocab_path = download_resource(&config.vocab_resource)?;
        let weights_path = download_resource(&config.model_resource)?;
        let merges_path = if let Some(merges_resource) = &config.merges_resource {
            Some(download_resource(merges_resource).expect("Failure downloading resource"))
        } else {
            None
        };
        let device = config.device;

        let tokenizer = TokenizerOption::from_file(
            config.model_type,
            vocab_path.to_str().unwrap(),
            merges_path.map(|path| path.to_str().unwrap()),
            config.lower_case,
            config.strip_accents,
            config.add_prefix_space,
        )?;
        let mut var_store = VarStore::new(device);
        let model_config = ConfigOption::from_file(config.model_type, config_path);
        let sequence_classifier =
            ZeroShotClassificationOption::new(config.model_type, &var_store.root(), &model_config);
        var_store.load(weights_path)?;
        Ok(ZeroShotClassificationModel {
            tokenizer,
            zero_shot_classifier: sequence_classifier,
            var_store,
        })
    }

    fn prepare_for_model(
        &self,
        inputs: &[&str],
        labels: &[&str],
        template: Option<Box<dyn Fn(&str) -> String>>,
        max_len: usize,
    ) -> Tensor {
        let label_sentences: Vec<String> = match template {
            Some(function) => labels.iter().map(|label| function(label)).collect(),
            None => labels
                .into_iter()
                .map(|label| format!("This example is {}.", label))
                .collect(),
        };

        let text_pair_list = inputs
            .into_iter()
            .cartesian_product(label_sentences.iter())
            .map(|(&s, label)| (s, label.as_str()))
            .collect();

        let tokenized_input: Vec<TokenizedInput> = self.tokenizer.encode_pair_list(
            text_pair_list,
            max_len,
            &TruncationStrategy::LongestFirst,
            0,
        );
        let max_len = tokenized_input
            .iter()
            .map(|input| input.token_ids.len())
            .max()
            .unwrap();
        let tokenized_input_tensors: Vec<tch::Tensor> =
            tokenized_input
                .iter()
                .map(|input| input.token_ids.clone())
                .map(|mut input| {
                    input.extend(vec![self.tokenizer.get_pad_id().expect(
                        "The Tokenizer used for zero shot classification should contain a PAD id"
                    ); max_len - input.len()]);
                    input
                })
                .map(|input| Tensor::of_slice(&(input)))
                .collect::<Vec<_>>();
        Tensor::stack(tokenized_input_tensors.as_slice(), 0).to(self.var_store.device())
    }

    /// Classify texts
    ///
    /// # Arguments
    ///
    /// * `input` - `&[&str]` Array of texts to classify.
    /// * `labels` - `&[&str]` Possible labels for the inputs.
    /// * `multilabel` - `bool` Flag indicating if 1 and exactly 1 label per sentence is true, or if 0, 1 or more labels can be true for each sentence.
    ///
    /// # Returns
    ///
    /// * `Vec<Vec<Label>>` containing a vector of labels for each input text
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::sequence_classification::SequenceClassificationModel;
    ///
    /// let sequence_classification_model =  SequenceClassificationModel::new(Default::default())?;
    /// let input = [
    ///     "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
    ///     "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
    ///     "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
    /// ];
    /// let output = sequence_classification_model.predict(&input);
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict(
        &self,
        inputs: &[&str],
        labels: &[&str],
        template: Option<Box<dyn Fn(&str) -> String>>,
        max_length: usize,
    ) -> Vec<Vec<Label>> {
        let num_inputs = inputs.len();
        let input_tensor = self.prepare_for_model(inputs, labels, template, max_length);
        let output = no_grad(|| {
            let output = self.sequence_classifier.forward_t(
                Some(input_tensor),
                None,
                None,
                None,
                None,
                false,
            );
            // output.softmax(-1, Kind::Float).detach().to(Device::Cpu)
        });
        let label_indices = output.as_ref().argmax(-1, true).squeeze1(1);
        let scores = output
            .gather(1, &label_indices.unsqueeze(-1), false)
            .squeeze1(1);
        let label_indices = label_indices.iter::<i64>().unwrap().collect::<Vec<i64>>();
        let scores = scores.iter::<f64>().unwrap().collect::<Vec<f64>>();

        let mut labels: Vec<Label> = vec![];
        for sentence_idx in 0..label_indices.len() {
            let label_string = self
                .label_mapping
                .get(&label_indices[sentence_idx])
                .unwrap()
                .clone();
            let label = Label {
                text: label_string,
                score: scores[sentence_idx],
                id: label_indices[sentence_idx],
                sentence: sentence_idx,
            };
            labels.push(label)
        }
        labels
    }
}
