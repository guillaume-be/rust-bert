use crate::pipelines::onnx::common::{get_input_output_mapping, InputOutputNameMapping};
use crate::pipelines::onnx::config::{
    ONNXEnvironmentConfig, ATTENTION_MASK_NAME, END_LOGITS, INPUT_EMBEDS, INPUT_IDS_NAME,
    LAST_HIDDEN_STATE, LOGITS, POSITION_IDS, START_LOGITS, TOKEN_TYPE_IDS,
};
use crate::pipelines::onnx::conversion::{array_to_ort, ort_tensor_to_tch, tch_tensor_to_ndarray};
use crate::RustBertError;
use ort::{Environment, Session};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tch::Tensor;

/// # ONNX Encoder model
/// Container for an ONNX encoder model and the corresponding session. Can be used individually for
/// pure-encoder models (e.g. BERT) or as part of encoder/decoder architectures.
pub struct ONNXEncoder {
    session: Session,
    name_mapping: InputOutputNameMapping,
}

impl ONNXEncoder {
    /// Create a new `ONNXEncoder`. Requires a pointer to the model file for
    /// the encoder, a reference to an environment and an ONNX environment configuration.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use ort::Environment;
    /// use rust_bert::pipelines::onnx::config::ONNXEnvironmentConfig;
    /// use rust_bert::pipelines::onnx::ONNXEncoder;
    /// use std::path::PathBuf;
    /// use std::sync::Arc;
    /// let environment = Arc::new(Environment::default());
    /// let onnx_config = ONNXEnvironmentConfig::default();
    /// let model_file = PathBuf::from("path/to/model.onnx");
    ///
    /// let encoder = ONNXEncoder::new(model_file, &environment, &onnx_config).unwrap();
    /// ```
    pub fn new(
        model_file: PathBuf,
        environment: &Arc<Environment>,
        onnx_config: &ONNXEnvironmentConfig,
    ) -> Result<Self, RustBertError> {
        let session = onnx_config
            .get_session_builder(environment)?
            .with_model_from_file(model_file)?;
        let name_mapping = get_input_output_mapping(&session);
        Ok(Self {
            session,
            name_mapping,
        })
    }

    /// Forward pass through the model.
    ///
    /// The outputs provided by the model depend on the underlying ONNX model and are all marked as optional to support a broad range of
    /// encoder stacks for multiple stacks. The end-user should extract the required output that is provided by the model exported.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `token_type_ids` - Optional segment id of shape (*batch size*, *sequence_length*). Convention is value of 0 for the first sentence (incl. *SEP*) and 1 for the second sentence. If None set to 0.
    /// * `position_ids` - Optional position ids of shape (*batch size*, *sequence_length*). If None, will be incremented from 0.
    /// * `input_embeds` - Optional pre-computed input embeddings of shape (*batch size*, *sequence_length*, *hidden_size*). If None, input ids must be provided (see `input_ids`)
    ///
    /// # Returns
    ///
    /// * `ONNXEncoderModelOutput` containing:
    ///   - `last_hidden_state` - Optional `Tensor` of shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `logits` - Optional `Tensor` of shape (*batch size*, *num_labels*)
    ///   - `start_logits` - Optional `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for start of the answer
    ///   - `end_logits` - Optional `Tensor` of shape (*batch size*, *sequence_length*) containing the logits for end of the answer
    ///   - `hidden_states` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///   - `attentions` - `Option<Vec<Tensor>>` of length *num_hidden_layers* with shape (*batch size*, *sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rust_bert::bert::{BertModel, BertConfig, BertEmbeddings};
    /// # use tch::{nn, Device, Tensor, no_grad, Kind};
    /// # use rust_bert::Config;
    /// # use std::path::Path;
    /// # let config_path = Path::new("path/to/config.json");
    /// # let device = Device::Cpu;
    /// # let vs = nn::VarStore::new(device);
    /// # let config = BertConfig::from_file(config_path);
    /// # let bert_model: BertModel<BertEmbeddings> = BertModel::new(&vs.root(), &config);
    /// let (batch_size, sequence_length) = (64, 128);
    /// let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let mask = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let token_type_ids = Tensor::zeros(&[batch_size, sequence_length], (Kind::Int64, device));
    /// let position_ids = Tensor::arange(sequence_length, (Kind::Int64, device))
    ///     .expand(&[batch_size, sequence_length], true);
    ///
    /// let model_output = no_grad(|| {
    ///     bert_model
    ///         .forward_t(
    ///             Some(&input_tensor),
    ///             Some(&mask),
    ///             Some(&token_type_ids),
    ///             Some(&position_ids),
    ///             None,
    ///             None,
    ///             None,
    ///             false,
    ///         )
    ///         .unwrap()
    /// });
    /// ```
    pub fn forward(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
    ) -> Result<ONNXEncoderModelOutput, RustBertError> {
        let mut input_dict = HashMap::new();
        if let Some(input_ids) = input_ids {
            input_dict.insert(INPUT_IDS_NAME, input_ids);
        }
        if let Some(attention_mask) = attention_mask {
            input_dict.insert(ATTENTION_MASK_NAME, attention_mask);
        }
        if let Some(token_type_ids) = token_type_ids {
            input_dict.insert(TOKEN_TYPE_IDS, token_type_ids);
        }
        if let Some(position_ids) = position_ids {
            input_dict.insert(POSITION_IDS, position_ids);
        }
        if let Some(input_embeds) = input_embeds {
            input_dict.insert(INPUT_EMBEDS, input_embeds);
        }

        let inputs_arrays = self
            .name_mapping
            .input_names
            .iter()
            .map(|input_name| {
                if let Some(tensor) = input_dict.remove(input_name.as_str()) {
                    tch_tensor_to_ndarray(tensor)
                } else {
                    Err(RustBertError::OrtError(format!(
                        "{input_name} not found but expected by model."
                    )))
                }
            })
            .collect::<Result<Vec<_>, RustBertError>>()?;

        let input_values = inputs_arrays
            .iter()
            .map(|array| array_to_ort(&self.session, array).unwrap())
            .collect::<Vec<_>>();

        let outputs = self.session.run(input_values)?;

        let last_hidden_state = self
            .name_mapping
            .output_names
            .get(LAST_HIDDEN_STATE)
            .map(|pos| ort_tensor_to_tch(&outputs[*pos]))
            .transpose()?;
        let logits = self
            .name_mapping
            .output_names
            .get(LOGITS)
            .map(|pos| ort_tensor_to_tch(&outputs[*pos]))
            .transpose()?;
        let start_logits = self
            .name_mapping
            .output_names
            .get(START_LOGITS)
            .map(|pos| ort_tensor_to_tch(&outputs[*pos]))
            .transpose()?;
        let end_logits = self
            .name_mapping
            .output_names
            .get(END_LOGITS)
            .map(|pos| ort_tensor_to_tch(&outputs[*pos]))
            .transpose()?;

        let (hidden_states, attentions) = if self.name_mapping.output_names.len() > 1 {
            let hidden_states = self
                .name_mapping
                .output_names
                .iter()
                .filter(|(name, _)| name.contains("hidden_states"))
                .map(|(_, position)| outputs.get(*position))
                .map(|array| array.map(|array_value| ort_tensor_to_tch(array_value).unwrap()))
                .collect::<Option<Vec<_>>>();

            let attentions = self
                .name_mapping
                .output_names
                .iter()
                .filter(|(name, _)| name.contains("attentions"))
                .map(|(_, position)| outputs.get(*position))
                .map(|array| array.map(|array_value| ort_tensor_to_tch(array_value).unwrap()))
                .collect::<Option<Vec<_>>>();
            (hidden_states, attentions)
        } else {
            (None, None)
        };
        Ok(ONNXEncoderModelOutput {
            last_hidden_state,
            logits,
            start_logits,
            end_logits,
            hidden_states,
            attentions,
        })
    }
}

/// # ONNX encoder model output.
/// The outputs provided by the model depend on the underlying ONNX model and are all marked as optional to support a broad range of
/// encoder stacks for multiple stacks. The end-user should extract the required output that is provided by the model exported.
pub struct ONNXEncoderModelOutput {
    /// Last hidden states, typically used by masked language model encoder models
    pub last_hidden_state: Option<Tensor>,
    /// logits, typically used by models with a sequence of classification head
    pub logits: Option<Tensor>,
    /// logits marking the start location of a span (e.g. for extractive question answering tasks)
    pub start_logits: Option<Tensor>,
    /// logits marking the end location of a span (e.g. for extractive question answering tasks)
    pub end_logits: Option<Tensor>,
    /// Hidden states for intermediate layers of the model
    pub hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for intermediate layers of the model
    pub attentions: Option<Vec<Tensor>>,
}
