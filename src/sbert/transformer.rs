use std::convert::TryInto;
use std::path::PathBuf;

use rust_tokenizers::tokenizer::{
    AlbertTokenizer, BertTokenizer, RobertaTokenizer, T5Tokenizer, Tokenizer, TruncationStrategy,
};
use rust_tokenizers::TokenizedInput;
use tch::{nn, Device, Tensor};

use crate::albert::AlbertModel;
use crate::bert::{BertEmbeddings, BertModel};
use crate::distilbert::DistilBertModel;
use crate::roberta::RobertaEmbeddings;
use crate::sbert::config::{SBertModelConfig, SBertTokenizerConfig};
use crate::t5::T5Model;
use crate::{Config, RustBertError};

pub trait SBertTransformer: private::Sealed + Sized {
    fn model_conf<P: Into<PathBuf>>(model_dir: P) -> SBertModelConfig {
        let model_dir = model_dir.into();
        let model_config_file = model_dir.join("config.json");
        SBertModelConfig::from_file(model_config_file)
    }

    fn tokenizer_conf<P: Into<PathBuf>>(model_dir: P) -> SBertTokenizerConfig {
        let model_dir = model_dir.into();
        let tokenizer_config_file = model_dir.join("sentence_bert_config.json");
        SBertTokenizerConfig::from_file(tokenizer_config_file)
    }

    fn from<P: Into<PathBuf>>(
        model_dir: P,
        device: Device,
    ) -> Result<(Self, nn::VarStore), RustBertError>;

    fn tokenize<S: AsRef<str>>(
        &self,
        inputs: &[S],
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> Vec<TokenizedInput>;

    fn forward(
        &self,
        tokens_ids: &Tensor,
        tokens_masks: &Tensor,
    ) -> Result<(Tensor, Option<Vec<Tensor>>), RustBertError>;
}

mod private {
    pub trait Sealed {}
    impl Sealed for super::UsingDistilBert {}
    impl Sealed for super::UsingBert {}
    impl Sealed for super::UsingRoberta {}
    impl Sealed for super::UsingAlbert {}
    impl Sealed for super::UsingT5 {}
}

pub struct UsingDistilBert {
    tokenizer: BertTokenizer,
    model: DistilBertModel,
}

impl SBertTransformer for UsingDistilBert {
    fn from<P: Into<PathBuf>>(
        model_dir: P,
        device: Device,
    ) -> Result<(Self, nn::VarStore), RustBertError> {
        let model_dir = model_dir.into();
        let conf_model = Self::model_conf(model_dir.clone());
        let conf_tokenizer = Self::tokenizer_conf(model_dir.clone());

        let mut var_store = nn::VarStore::new(device);
        let model = DistilBertModel::new(&var_store.root(), &conf_model.try_into()?);

        let weights_file = model_dir.join("rust_model.ot");
        var_store.load(weights_file)?;

        let tokenizer = BertTokenizer::from_file(
            &model_dir.join("vocab.txt").to_string_lossy(),
            conf_tokenizer.do_lower_case,
            false,
        )?;

        Ok((Self { tokenizer, model }, var_store))
    }

    fn tokenize<S: AsRef<str>>(
        &self,
        inputs: &[S],
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> Vec<TokenizedInput> {
        self.tokenizer
            .encode_list(inputs, max_len, truncation_strategy, stride)
    }

    fn forward(
        &self,
        tokens_ids: &Tensor,
        tokens_masks: &Tensor,
    ) -> Result<(Tensor, Option<Vec<Tensor>>), RustBertError> {
        let output = tch::no_grad(|| {
            self.model
                .forward_t(Some(tokens_ids), Some(tokens_masks), None, false)
        })?;
        Ok((output.hidden_state, output.all_attentions))
    }
}

pub struct UsingBert {
    tokenizer: BertTokenizer,
    model: BertModel<BertEmbeddings>,
}

impl SBertTransformer for UsingBert {
    fn from<P: Into<PathBuf>>(
        model_dir: P,
        device: Device,
    ) -> Result<(Self, nn::VarStore), RustBertError> {
        let model_dir = model_dir.into();
        let conf_model = Self::model_conf(model_dir.clone());
        let conf_tokenizer = Self::tokenizer_conf(model_dir.clone());

        let mut var_store = nn::VarStore::new(device);
        let model = BertModel::<BertEmbeddings>::new(&var_store.root(), &conf_model.try_into()?);

        let weights_file = model_dir.join("rust_model.ot");
        var_store.load(weights_file)?;

        let tokenizer = BertTokenizer::from_file(
            &model_dir.join("vocab.txt").to_string_lossy(),
            conf_tokenizer.do_lower_case,
            false,
        )?;

        Ok((Self { tokenizer, model }, var_store))
    }

    fn tokenize<S: AsRef<str>>(
        &self,
        inputs: &[S],
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> Vec<TokenizedInput> {
        self.tokenizer
            .encode_list(inputs, max_len, truncation_strategy, stride)
    }

    fn forward(
        &self,
        tokens_ids: &Tensor,
        tokens_masks: &Tensor,
    ) -> Result<(Tensor, Option<Vec<Tensor>>), RustBertError> {
        let output = tch::no_grad(|| {
            self.model.forward_t(
                Some(tokens_ids),
                Some(tokens_masks),
                None,
                None,
                None,
                None,
                None,
                false,
            )
        })?;
        Ok((output.hidden_state, output.all_attentions))
    }
}

pub struct UsingRoberta {
    tokenizer: RobertaTokenizer,
    model: BertModel<RobertaEmbeddings>,
}

impl SBertTransformer for UsingRoberta {
    fn from<P: Into<PathBuf>>(
        model_dir: P,
        device: Device,
    ) -> Result<(Self, nn::VarStore), RustBertError> {
        let model_dir = model_dir.into();
        let conf_model = Self::model_conf(model_dir.clone());
        let conf_tokenizer = Self::tokenizer_conf(model_dir.clone());

        let mut var_store = nn::VarStore::new(device);
        let model = BertModel::<RobertaEmbeddings>::new_with_optional_pooler(
            &var_store.root(),
            &conf_model.try_into()?,
            false,
        );

        let weights_file = model_dir.join("rust_model.ot");
        var_store.load(weights_file)?;

        let tokenizer = RobertaTokenizer::from_file(
            &model_dir.join("vocab.json").to_string_lossy(),
            &model_dir.join("merges.txt").to_string_lossy(),
            conf_tokenizer.do_lower_case,
            false,
        )?;

        Ok((Self { tokenizer, model }, var_store))
    }

    fn tokenize<S: AsRef<str>>(
        &self,
        inputs: &[S],
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> Vec<TokenizedInput> {
        self.tokenizer
            .encode_list(inputs, max_len, truncation_strategy, stride)
    }

    fn forward(
        &self,
        tokens_ids: &Tensor,
        tokens_masks: &Tensor,
    ) -> Result<(Tensor, Option<Vec<Tensor>>), RustBertError> {
        let output = tch::no_grad(|| {
            self.model.forward_t(
                Some(tokens_ids),
                Some(tokens_masks),
                None,
                None,
                None,
                None,
                None,
                false,
            )
        })?;
        Ok((output.hidden_state, output.all_attentions))
    }
}

pub struct UsingAlbert {
    tokenizer: AlbertTokenizer,
    model: AlbertModel,
}

impl SBertTransformer for UsingAlbert {
    fn from<P: Into<PathBuf>>(
        model_dir: P,
        device: Device,
    ) -> Result<(Self, nn::VarStore), RustBertError> {
        let model_dir = model_dir.into();
        let conf_model = Self::model_conf(model_dir.clone());
        let conf_tokenizer = Self::tokenizer_conf(model_dir.clone());

        let mut var_store = nn::VarStore::new(device);
        let model = AlbertModel::new(&var_store.root(), &conf_model.try_into()?);

        let weights_file = model_dir.join("rust_model.ot");
        var_store.load(weights_file)?;

        let tokenizer = AlbertTokenizer::from_file(
            &model_dir.join("spiece.model").to_string_lossy(),
            conf_tokenizer.do_lower_case,
            false,
        )?;

        Ok((Self { tokenizer, model }, var_store))
    }

    fn tokenize<S: AsRef<str>>(
        &self,
        inputs: &[S],
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> Vec<TokenizedInput> {
        self.tokenizer
            .encode_list(inputs, max_len, truncation_strategy, stride)
    }

    fn forward(
        &self,
        tokens_ids: &Tensor,
        tokens_masks: &Tensor,
    ) -> Result<(Tensor, Option<Vec<Tensor>>), RustBertError> {
        let output = tch::no_grad(|| {
            self.model.forward_t(
                Some(tokens_ids),
                Some(tokens_masks),
                None,
                None,
                None,
                false,
            )
        })?;
        Ok((
            output.hidden_state,
            // Average attentions of each inner group
            output.all_attentions.map(|attentions| {
                attentions
                    .into_iter()
                    .map(|tensors| {
                        let num_inner_groups = tensors.len() as f64;
                        tensors.into_iter().sum::<Tensor>() / num_inner_groups
                    })
                    .collect()
            }),
        ))
    }
}

pub struct UsingT5 {
    tokenizer: T5Tokenizer,
    model: T5Model,
}

impl SBertTransformer for UsingT5 {
    fn from<P: Into<PathBuf>>(
        model_dir: P,
        device: Device,
    ) -> Result<(Self, nn::VarStore), RustBertError> {
        let model_dir = model_dir.into();
        let conf_model = Self::model_conf(model_dir.clone());
        let conf_tokenizer = Self::tokenizer_conf(model_dir.clone());

        let output_attentions = conf_model.output_attentions();
        let mut var_store = nn::VarStore::new(device);
        let model = T5Model::new(
            &var_store.root(),
            &conf_model.try_into()?,
            output_attentions,
            false,
        );

        let weights_file = model_dir.join("rust_model.ot");
        var_store.load(weights_file)?;

        let tokenizer = T5Tokenizer::from_file(
            &model_dir.join("tokenizer.json").to_string_lossy(),
            conf_tokenizer.do_lower_case,
        )?;

        Ok((Self { tokenizer, model }, var_store))
    }

    fn tokenize<S: AsRef<str>>(
        &self,
        inputs: &[S],
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> Vec<TokenizedInput> {
        self.tokenizer
            .encode_list(inputs, max_len, truncation_strategy, stride)
    }

    fn forward(
        &self,
        tokens_ids: &Tensor,
        tokens_masks: &Tensor,
    ) -> Result<(Tensor, Option<Vec<Tensor>>), RustBertError> {
        let output = tch::no_grad(|| {
            self.model.forward_t(
                Some(tokens_ids),
                Some(tokens_masks),
                None,
                None,
                None,
                None,
                None,
                None,
                false,
            )
        });
        Ok((
            output.encoder_hidden_state.unwrap(),
            output.all_encoder_attentions,
        ))
    }
}
