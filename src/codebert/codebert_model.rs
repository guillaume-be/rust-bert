// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::bert::{BertConfig, BertModel};
use crate::common::error::RustBertError;
use crate::pipelines::common::TokenizerOption;
use crate::pipelines::sequence_classification::SequenceClassificationConfig;
use crate::resources::ResourceProvider;
use crate::roberta::{
    RobertaEmbeddings, RobertaForMaskedLM, RobertaForMultipleChoice, RobertaForQuestionAnswering,
    RobertaForSequenceClassification, RobertaForTokenClassification,
};
use crate::Config;
use rust_tokenizers::tokenizer::TruncationStrategy;
use rust_tokenizers::TokenizedInput;
use std::borrow::Borrow;
use tch::nn::VarStore;
use tch::{nn, no_grad, Device, Tensor};

/// # CODEBERT Pretrained model weight files
pub struct CodeBertModelResources;

/// # CODEBERT Pretrained model config files
pub struct CodeBertConfigResources;

/// # CODEBERT Pretrained model vocab files
pub struct CodeBertVocabResources;

/// # CODEBERT Pretrained model merges files
pub struct CodeBertMergesResources;
/// CODEBERT has same basic module implementation as ROBERTA
pub type CodeBertConfig = BertConfig;
pub type CodeBertForFeatureExtractionConfig = SequenceClassificationConfig;
pub type CodeBertEmbeddings = RobertaEmbeddings;
pub type CodeBertForMaskedLM = RobertaForMaskedLM;
pub type CodeBertForMultipleChoice = RobertaForMultipleChoice;
pub type CodeBertForSequenceClassification = RobertaForSequenceClassification;
pub type CodeBertForSentenceEmbeddings = BertModel<CodeBertEmbeddings>;
pub type CodeBertForTokenClassification = RobertaForTokenClassification;
pub type CodeBertForQuestionAnswering = RobertaForQuestionAnswering;
impl CodeBertModelResources {
    /// Shared under Apache 2.0 license by the Microsoft team at <https://github.com/microsoft/CodeBERT>. Modified with conversion to C-array format.
    pub const CODEBERT: (&'static str, &'static str) = (
        "codebert/model",
        "https://huggingface.co/microsoft/codebert-base/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the Hugging Face Inc. team at <https://huggingface.co/huggingface/CodeBERTa-language-id>. Modified with conversion to C-array format.
    pub const CODEBERTA_LANG: (&'static str, &'static str) = (
        "codeberta-language-id/model",
        "https://huggingface.co/huggingface/CodeBERTa-language-id/resolve/main/rust_model.ot",
    );
    /// Shared under Apache 2.0 license by the Microsoft team at <https://github.com/microsoft/CodeBERT>. Modified with conversion to C-array format.
    pub const CODEBERT_MLM: (&'static str, &'static str) = (
        "codebert-mlm/model",
        "https://huggingface.co/microsoft/codebert-base-mlm/resolve/main/rust_model.ot",
    );
}

impl CodeBertConfigResources {
    /// Shared under Apache 2.0 license by the Microsoft team at <https://github.com/microsoft/CodeBERT>. Modified with conversion to C-array format.
    pub const CODEBERT: (&'static str, &'static str) = (
        "codebert/config",
        "https://huggingface.co/microsoft/codebert-base/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the Hugging Face Inc. team at <https://huggingface.co/huggingface/CodeBERTa-language-id>. Modified with conversion to C-array format.
    pub const CODEBERTA_LANG: (&'static str, &'static str) = (
        "codeberta-language-id/config",
        "https://huggingface.co/huggingface/CodeBERTa-language-id/resolve/main/config.json",
    );
    /// Shared under Apache 2.0 license by the Microsoft team at <https://github.com/microsoft/CodeBERT>. Modified with conversion to C-array format.
    pub const CODEBERT_MLM: (&'static str, &'static str) = (
        "codebert-mlm/model",
        "https://huggingface.co/microsoft/codebert-base-mlm/resolve/main/config.json",
    );
}

impl CodeBertVocabResources {
    /// Shared under Apache 2.0 license by the Microsoft team at <https://github.com/microsoft/CodeBERT>. Modified with conversion to C-array format.
    pub const CODEBERT: (&'static str, &'static str) = (
        "codebert/vocab",
        "https://huggingface.co/microsoft/codebert-base/resolve/main/vocab.json",
    );
    /// Shared under Apache 2.0 license by the Hugging Face Inc. team at <https://huggingface.co/huggingface/CodeBERTa-language-id>. Modified with conversion to C-array format.
    pub const CODEBERTA_LANG: (&'static str, &'static str) = (
        "codeberta-language-id/vocab",
        "https://huggingface.co/huggingface/CodeBERTa-language-id/resolve/main/vocab.json",
    );
    /// Shared under Apache 2.0 license by the Microsoft team at <https://github.com/microsoft/CodeBERT>. Modified with conversion to C-array format.
    pub const CODEBERT_MLM: (&'static str, &'static str) = (
        "codebert-mlm/model",
        "https://huggingface.co/microsoft/codebert-base-mlm/resolve/main/vocab.json",
    );
}

impl CodeBertMergesResources {
    /// Shared under Apache 2.0 license by the Microsoft team at <https://github.com/microsoft/CodeBERT>. Modified with conversion to C-array format.
    pub const CODEBERT: (&'static str, &'static str) = (
        "codebert/merges",
        "https://huggingface.co/microsoft/codebert-base/resolve/main/merges.txt",
    );
    /// Shared under Apache 2.0 license by the Hugging Face Inc. team at <https://huggingface.co/huggingface/CodeBERTa-language-id>. Modified with conversion to C-array format.
    pub const CODEBERTA_LANG: (&'static str, &'static str) = (
        "codeberta-language-id/merges",
        "https://huggingface.co/huggingface/CodeBERTa-language-id/resolve/main/merges.txt",
    );
    /// Shared under Apache 2.0 license by the Microsoft team at <https://github.com/microsoft/CodeBERT>. Modified with conversion to C-array format.
    pub const CODEBERT_MLM: (&'static str, &'static str) = (
        "codebert-mlm/model",
        "https://huggingface.co/microsoft/codebert-base-mlm/resolve/main/merges.txt",
    );
}

pub struct CodeBertForFeatureExtraction {
    codebert: BertModel<CodeBertEmbeddings>,
}

impl CodeBertForFeatureExtraction {
    pub fn new<'p, P>(p: P, config: &BertConfig) -> CodeBertForFeatureExtraction
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let codebert = BertModel::<CodeBertEmbeddings>::new_with_optional_pooler(p, config, false);

        CodeBertForFeatureExtraction { codebert }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        encoder_mask: Option<&Tensor>,
        train: bool,
    ) -> CodeBertFeatureExtractionOutput {
        let base_model_output = self
            .codebert
            .forward_t(
                input_ids,
                mask,
                token_type_ids,
                position_ids,
                input_embeds,
                encoder_hidden_states,
                encoder_mask,
                train,
            )
            .map(|transformer_output| {
                (
                    transformer_output.hidden_state,
                    transformer_output.all_attentions,
                )
            });
        let (tokens_embeddings, all_attentions) = base_model_output.unwrap();
        CodeBertFeatureExtractionOutput {
            hidden_states: tokens_embeddings,
            all_attentions: all_attentions,
        }
    }
}
pub struct CodeBertFeatureExtractionOutput {
    /// Hidden states for all intermediate layers
    pub hidden_states: Tensor,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}
pub struct CodeBertForFeatureExtractionModel {
    tokenizer: TokenizerOption,
    codebert: CodeBertForFeatureExtraction,
    var_store: VarStore,
    max_length: usize,
}
impl CodeBertForFeatureExtractionModel {
    /// Build a new `CodeBertForFeatureExtractionModel`
    ///
    /// # Arguments
    ///
    /// * `config` - `CodeBertForFeatureExtractionConfig` object containing the resource references (model, vocabulary, configuration) and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::common::ModelType;
    /// use rust_bert::codebert::{
    /// CodeBertConfigResources, CodeBertForFeatureExtractionConfig,CodeBertForFeatureExtractionModel,
    /// CodeBertMergesResources, CodeBertModelResources, CodeBertVocabResources,
    /// };
    /// use rust_bert::resources::RemoteResource;
    /// let config = CodeBertForFeatureExtractionConfig::new(
    ///     ModelType::CodeBert,
    ///     RemoteResource::from_pretrained(CodeBertModelResources::CODEBERT),
    ///     None,
    ///     RemoteResource::from_pretrained(CodeBertConfigResources::CODEBERT),
    ///     RemoteResource::from_pretrained(CodeBertVocabResources::CODEBERT),
    ///     RemoteResource::from_pretrained(CodeBertMergesResources::CODEBERT),
    ///     true,
    ///     None,
    ///     None,
    /// );
    /// let feature_extraction_model = CodeBertForFeatureExtractionModel::new(config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        config: CodeBertForFeatureExtractionConfig,
    ) -> Result<CodeBertForFeatureExtractionModel, RustBertError> {
        let config_path = config.config_resource.get_local_path()?;
        let vocab_path = config.vocab_resource.get_local_path()?;
        let weights_path = if config.model_local_resource.is_none() {
            config.model_resource.unwrap().get_local_path()?
        } else {
            config.model_local_resource.unwrap().get_local_path()?
        };
        let merges_path = if let Some(merges_resource) = &config.merges_resource {
            Some(merges_resource.get_local_path()?)
        } else {
            None
        };
        let device = config.device;

        let tokenizer = TokenizerOption::from_file(
            config.model_type,
            vocab_path.to_str().unwrap(),
            merges_path.as_deref().map(|path| path.to_str().unwrap()),
            config.lower_case,
            config.strip_accents,
            config.add_prefix_space,
        )?;
        let mut var_store = VarStore::new(device);
        let model_config = CodeBertConfig::from_file(config_path);
        let max_length = Some(model_config.max_position_embeddings)
            .map(|v| v as usize)
            .unwrap_or(usize::MAX);
        let codebert = CodeBertForFeatureExtraction::new(&var_store.root(), &model_config);
        var_store.load(weights_path)?;
        Ok(CodeBertForFeatureExtractionModel {
            tokenizer,
            codebert,
            var_store,
            max_length,
        })
    }

    fn prepare_for_model<'a, S>(&self, input: S) -> Tensor
    where
        S: AsRef<[&'a str]>,
    {
        let tokenized_input: Vec<TokenizedInput> = self.tokenizer.encode_list(
            input.as_ref(),
            self.max_length,
            &TruncationStrategy::LongestFirst,
            0,
        );
        let max_len = tokenized_input
            .iter()
            .map(|input| input.token_ids.len())
            .max()
            .unwrap();
        let pad_id = self
            .tokenizer
            .get_pad_id()
            .expect("The Tokenizer used for sequence classification should contain a PAD id");
        let tokenized_input_tensors: Vec<tch::Tensor> = tokenized_input
            .into_iter()
            .map(|mut input| {
                input.token_ids.resize(max_len, pad_id);
                Tensor::of_slice(&(input.token_ids))
            })
            .collect::<Vec<_>>();
        Tensor::stack(tokenized_input_tensors.as_slice(), 0).to(self.var_store.device())
    }

    /// Extract  feature of texts
    ///
    /// # Arguments
    ///
    /// * `input` - `&[&str]` Array of texts to extract.
    ///
    /// # Returns
    ///
    /// * `Vec<Tensor>` containing hidden_states for input texts
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::common::ModelType;
    /// use rust_bert::codebert::{
    /// CodeBertConfigResources, CodeBertForFeatureExtractionConfig,CodeBertForFeatureExtractionModel,
    /// CodeBertMergesResources, CodeBertModelResources, CodeBertVocabResources,
    /// };
    /// use rust_bert::resources::RemoteResource;
    ///
    /// let config = CodeBertForFeatureExtractionConfig::new(
    ///     ModelType::CodeBert,
    ///     RemoteResource::from_pretrained(CodeBertModelResources::CODEBERT),
    ///     None,
    ///     RemoteResource::from_pretrained(CodeBertConfigResources::CODEBERT),
    ///     RemoteResource::from_pretrained(CodeBertVocabResources::CODEBERT),
    ///     RemoteResource::from_pretrained(CodeBertMergesResources::CODEBERT),
    ///     true,
    ///     None,
    ///     None,
    /// );
    /// let feature_extraction_model = CodeBertForFeatureExtractionModel::new(config)?;
    /// //    Define input
    /// let input = ["this is an example sentence", "each sentence is converted"];
    /// //    Run model
    /// let output = feature_extraction_model.predict(&input);
    /// for hidden_states in output {
    ///     println!("{:?}", hidden_states);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict<'a, S>(&self, input: S) -> Vec<Tensor>
    where
        S: AsRef<[&'a str]>,
    {
        let input_tensor = self.prepare_for_model(input.as_ref());
        let output = no_grad(|| {
            let output = self
                .codebert
                .forward_t(
                    Some(&input_tensor),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                )
                .hidden_states;
            output.to(Device::Cpu)
        });

        let mut features: Vec<Tensor> = vec![];
        for idx in 0..output.size()[0] {
            features.push(output.get(idx));
        }
        features
    }
}
