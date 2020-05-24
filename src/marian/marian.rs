use crate::bart::{BartModel, BartConfig};
use tch::{Tensor, nn};
use std::borrow::BorrowMut;
use crate::pipelines::generation::LMHeadModel;
use tch::nn::Init;

/// # Marian Model for conditional generation
/// Marian model with a vocabulary decoding head
/// It is made of the following blocks:
/// - `base_model`: `BartModel` Base BART model
/// - `linear`: Linear layer with bias tied to the weights of the token id embeddings
pub struct MarianForConditionalGeneration {
    base_model: BartModel,
    final_logits_bias: Tensor,
}

impl MarianForConditionalGeneration {
    /// Build a new `MarianForConditionalGeneration`
    ///
    /// # Arguments
    ///
    /// * `p` - Variable store path for the root of the BART model
    /// * `config` - `BartConfig` object defining the model architecture
    /// * `generation_mode` - flag indicating if the model should run in generation mode (a decoder start token must then be provided)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use tch::{nn, Device};
    /// use rust_bert::Config;
    /// use std::path::Path;
    /// use rust_bert::bart::{BartConfig, BartForConditionalGeneration};
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let device = Device::Cpu;
    /// let p = nn::VarStore::new(device);
    /// let config = BartConfig::from_file(config_path);
    /// let generation_mode = true;
    /// let bart: BartForConditionalGeneration = BartForConditionalGeneration::new(&(&p.root() / "bart"), &config, generation_mode);
    /// ```
    ///
    pub fn new(p: &nn::Path, config: &BartConfig, generation_mode: bool) -> MarianForConditionalGeneration {
        let base_model = BartModel::new(&(p / "model"), config, generation_mode);
        let final_logits_bias = p.var("final_logits_bias", &[1, config.vocab_size], Init::Const(0.));
        MarianForConditionalGeneration { base_model, final_logits_bias }
    }

    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *source_sequence_length*). Must be provided when not running in generation mode
    /// * `attention_mask` - Optional attention mask of shape (*batch size*, *source_sequence_length*) for the encoder positions. Positions with a mask with value 0 will be masked.
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialiazed with a BOS token)
    /// * `decoder_attention_mask` - Optional attention mask of shape (*batch size*, *target_sequence_length*) for the decoder positions. Positions with a mask with value 0 will be masked.
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    /// # Returns
    ///
    /// * `lm_logits` - `Tensor` of shape (*batch size*, *target_sequence_length*, *vocab_size*) representing the logits for each vocab item and position
    /// * `encoder_hidden_states` - `Tensor` of shape (*batch size*, *source_sequence_length*, *hidden_size*) representing the activations of the last encoder hidden state
    /// * `all_encoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    /// * `all_encoder_attentions` - `Option<Vec<Tensor>>` of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*)
    /// * `all_decoder_hidden_states` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    /// * `all_decoder_attentions` - `Option<Vec<Tensor>>` of length *num_decoder_layers* with shape (*batch size*, *target_sequence_length*, *hidden_size*)
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::bart::{BartConfig, BartForConditionalGeneration};
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = BartConfig::from_file(config_path);
    ///# let mut bart_model: BartForConditionalGeneration = BartForConditionalGeneration::new(&vs.root(), &config, false);
    ///  let (batch_size, source_sequence_length, target_sequence_length) = (64, 128, 56);
    ///  let input_tensor = Tensor::rand(&[batch_size, source_sequence_length], (Int64, device));
    ///  let target_tensor = Tensor::rand(&[batch_size, target_sequence_length], (Int64, device));
    ///  let encoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///  let decoder_attention_mask = Tensor::ones(&[batch_size, source_sequence_length], (Int64, device));
    ///
    ///  let (decoder_output, encoder_hidden_states,
    ///       all_encoder_hidden_states, all_encoder_attentions,
    ///       all_decoder_hidden_states, all_decoder_attentions) = no_grad(|| {
    ///    bart_model
    ///         .forward_t(Some(&input_tensor),
    ///                    Some(&encoder_attention_mask),
    ///                    None,
    ///                    Some(&target_tensor),
    ///                    Some(&decoder_attention_mask),
    ///                    false)
    ///    });
    ///
    /// ```
    ///
    pub fn forward_t(&mut self,
                     input_ids: Option<&Tensor>,
                     attention_mask: Option<&Tensor>,
                     encoder_outputs: Option<(Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>)>,
                     decoder_input_ids: Option<&Tensor>,
                     decoder_attention_mask: Option<&Tensor>,
                     train: bool)
                     -> (Tensor, Tensor,
                         Option<Vec<Tensor>>, Option<Vec<Tensor>>,
                         Option<Vec<Tensor>>, Option<Vec<Tensor>>)
    {
        let (decoder_outputs, encoder_hidden_states, _,
            all_decoder_hidden_states, all_decoder_attentions,
            all_encoder_hidden_states, all_encoder_attentions) =
            self.borrow_mut().base_model.forward_t(input_ids, attention_mask, decoder_input_ids, encoder_outputs, decoder_attention_mask, train);

        let lm_logits = decoder_outputs.linear::<Tensor>(&self.base_model.embeddings.ws, None);
        (lm_logits, encoder_hidden_states,
         all_decoder_hidden_states, all_decoder_attentions,
         all_encoder_hidden_states, all_encoder_attentions)
    }

    pub(crate) fn get_base_model(&mut self) -> &mut BartModel { &mut self.base_model }

    pub fn encode(&mut self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        let (encoder_hidden_states, _, _) = self.base_model.encoder.forward_t(input_ids, attention_mask, &self.base_model.embeddings, false);
        encoder_hidden_states
    }

    /// Resets the decoder cached keys and values. Should be run for every new generation using the model.
    pub fn reset_cache(&mut self) {
        self.get_base_model().reset_cache()
    }
}

impl LMHeadModel for MarianForConditionalGeneration {
    /// Forward pass through the model
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Optional input tensor of shape (*batch size*, *sequence_length*). If None, pre-computed embeddings must be provided (see `input_embeds`)
    /// * `layer_past` - Unused for BART
    /// * `attention_mask` - Optional mask of shape (*batch size*, *sequence_length*). Masked position have value 0, non-masked value 1. If None set to 1
    /// * `input_embeds` - Unused for BART
    /// * `token_type_ids` - Unused for BART
    /// * `position_ids` - Unused for BART
    /// * `encoder_outputs` - Optional tuple made of a tensor of shape (*batch size*, *source_sequence_length*, *encoder_hidden_dim*) and optional vectors of tensors of length *num_encoder_layers* with shape (*batch size*, *source_sequence_length*, *hidden_size*).
    /// These correspond to the encoder last hidden state and optional hidden states/attention weights for encoder layers. When provided, the encoder hidden state will not be recalculated. Useful for generation tasks.
    /// * `decoder_input_ids` - Optional input tensor of shape (*batch size*, *target_sequence_length*). Must be provided when running in generation mode (e.g. initialiazed with a BOS token)
    /// * `train` - boolean flag to turn on/off the dropout layers in the model. Should be set to false for inference.
    ///
    ///
    /// # Returns
    ///
    /// * `lm_logits` - `Tensor` of shape (*batch size*, *sequence_length*, *vocab_size*) representing the logits for each vocab item and position
    /// * `past` - None
    /// * `encoder_hidden_states` - `Option<Tensor>` Hidden states for the encoder
    /// * `hidden_states` - None
    /// * `attentions` - None
    ///
    /// # Example
    ///
    /// ```no_run
    ///# use tch::{nn, Device, Tensor, no_grad};
    ///# use rust_bert::Config;
    ///# use std::path::Path;
    ///# use tch::kind::Kind::{Int64, Double};
    /// use rust_bert::gpt2::{Gpt2Config, GPT2LMHeadModel};
    /// use rust_bert::pipelines::generation::LMHeadModel;
    ///# let config_path = Path::new("path/to/config.json");
    ///# let vocab_path = Path::new("path/to/vocab.txt");
    ///# let device = Device::Cpu;
    ///# let vs = nn::VarStore::new(device);
    ///# let config = Gpt2Config::from_file(config_path);
    ///# let mut gpt2_model: GPT2LMHeadModel = GPT2LMHeadModel::new(&vs.root(), &config);
    ///  let (batch_size, sequence_length, past_sequence_length) = (64, 128, 56);
    ///  let input_tensor = Tensor::rand(&[batch_size, sequence_length], (Int64, device));
    ///  let mut past: Vec<Tensor> = Vec::with_capacity(config.n_layer as usize);
    ///  for _ in 0..config.n_layer as usize {
    ///    past.push(Tensor::rand(&[2, batch_size, config.n_head, past_sequence_length, config.n_embd / config.n_head], (Double, device)))
    /// }
    ///  let attention_mask = Tensor::zeros(&[batch_size, sequence_length], (Int64, device));
    ///  let token_type_ids = Tensor::ones(&[batch_size, sequence_length], (Int64, device));
    ///  let position_ids = Tensor::arange(sequence_length, (Int64, device)).expand(&[batch_size, sequence_length], true);
    ///
    ///  let (output, encoder_hidden_states, _, hidden_states, attentions) = no_grad(|| {
    ///    gpt2_model
    ///         .forward_t(&Some(input_tensor),
    ///                    &Some(past),
    ///                    &Some(attention_mask),
    ///                    &Some(token_type_ids),
    ///                    &Some(position_ids),
    ///                    &None,
    ///                    None,
    ///                    &None,
    ///                    false).unwrap()
    ///    });
    ///
    /// ```
    ///
    fn forward_t(&mut self,
                 input_ids: &Option<Tensor>,
                 _layer_past: &Option<Vec<Tensor>>,
                 attention_mask: &Option<Tensor>,
                 _token_type_ids: &Option<Tensor>,
                 _position_ids: &Option<Tensor>,
                 _input_embeds: &Option<Tensor>,
                 encoder_outputs: Option<&Tensor>,
                 decoder_input_ids: &Option<Tensor>,
                 train: bool) -> Result<(Tensor, Option<Tensor>, Option<Vec<Tensor>>, Option<Vec<Tensor>>, Option<Vec<Tensor>>), &'static str> {
        let (decoder_output, encoder_hidden_states, _, _, _, _, _) = self.base_model.forward_t(input_ids.as_ref(),
                                                                                               attention_mask.as_ref(),
                                                                                               decoder_input_ids.as_ref(),
                                                                                               Some((encoder_outputs.as_ref().unwrap().copy(), None, None)),
                                                                                               None,
                                                                                               train);

        let lm_logits = decoder_output.linear::<Tensor>(&self.base_model.embeddings.ws, None) + &self.final_logits_bias;
        
        Ok((lm_logits, Some(encoder_hidden_states), None, None, None))
    }
}