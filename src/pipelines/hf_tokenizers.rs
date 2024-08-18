/// # Support for [tokenizers](https://github.com/huggingface/tokenizers)
///
/// This module implements interface methods to allow loading tokenizers trained and implemented with
/// the [Tokenizers](https://github.com/huggingface/tokenizers) crate. While the functionality of these tokenizers
/// is expected to be identical to the default [rust-tokenizers](https://github.com/guillaume-be/rust-tokenizers) used
/// in this crate, the implementation and input file format differs.
///
/// Because some of the logic related to the special token handling is implemented at the Python level using the rust bindings,
/// the proposed implementation requires two files to be provided:
/// - `tokenizer.json` containing the tokenizer model, pre- and post-processing options and vocabulary
/// - `special_token_map.json` containing a mapping of the special tokens used by the model (e.g. BOS and CLS values)
use crate::RustBertError;
use rust_tokenizers::{
    Mask, Offset, OffsetSize, TokenIdsWithOffsets, TokenizedInput, TokensWithOffsets,
};
use serde::{de, Deserialize, Deserializer};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use tokenizers::tokenizer::Tokenizer as HFBaseTokenizer;
use tokenizers::{AddedToken, EncodeInput, Encoding, InputSequence};

impl From<tokenizers::tokenizer::Error> for RustBertError {
    fn from(error: tokenizers::tokenizer::Error) -> Self {
        RustBertError::TokenizerError(error.to_string())
    }
}

/// Container for a special token map to be deserialized from a `special_token_map.json`
#[derive(Debug, Default, Clone, Deserialize)]
pub struct SpecialTokenMap {
    /// Unknown token (must be provided for all tokenizers)
    pub unk_token: String,
    /// Optional padding token
    #[serde(default)]
    #[serde(deserialize_with = "string_or_added_token_struct")]
    pub pad_token: Option<String>,
    /// Optional bos token
    #[serde(default)]
    #[serde(deserialize_with = "string_or_added_token_struct")]
    pub bos_token: Option<String>,
    /// Optional sep token
    #[serde(default)]
    #[serde(deserialize_with = "string_or_added_token_struct")]
    pub sep_token: Option<String>,
    /// Optional cls token
    #[serde(default)]
    #[serde(deserialize_with = "string_or_added_token_struct")]
    pub cls_token: Option<String>,
    /// Optional eos token
    #[serde(default)]
    #[serde(deserialize_with = "string_or_added_token_struct")]
    pub eos_token: Option<String>,
    /// Optional mask token
    #[serde(default)]
    #[serde(deserialize_with = "string_or_added_token_struct")]
    pub mask_token: Option<String>,
    /// Optional additional special tokens
    pub additional_special_tokens: Option<HashSet<String>>,
}

/// Deserialization utility function for `special_token_map.json` to read nested special tokens structure
fn string_or_added_token_struct<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringOrStruct;

    impl<'de> de::Visitor<'de> for StringOrStruct {
        type Value = Option<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("string or map")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Some(value.to_string()))
        }

        fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
        where
            M: de::MapAccess<'de>,
        {
            let mut value = None;
            while let Some(key) = map.next_key::<String>()? {
                if key == "content" {
                    value = Some(map.next_value::<String>()?);
                } else {
                    _ = map.next_value::<String>();
                }
            }
            Ok(value)
        }
    }
    deserializer.deserialize_any(StringOrStruct)
}

/// Base class for a tokenizer from the Tokenizers library
pub struct HFTokenizer {
    /// Base tokenizer object
    tokenizer: HFBaseTokenizer,
    /// Special token map
    pub(crate) special_token_map: SpecialTokenMap,
}

impl HFTokenizer {
    /// Create a new tokenizer from a file.
    ///
    /// # Arguments
    /// - `tokenizer_file` path to location containing the tokenizer model, pre- and post-processing options and vocabulary
    /// - `special_token_map` path to location containing a mapping of the special tokens used by the model (e.g. BOS and CLS values)
    ///
    /// # Returns
    /// - Wrapper around a tokenizer that can be loaded in a `TokenizerOption` in this crate
    ///
    /// # Example
    ///
    /// ```no_run
    ///  # fn main() -> anyhow::Result<()> {
    /// use rust_bert::pipelines::hf_tokenizers::HFTokenizer;
    /// use std::path::PathBuf;
    /// let tokenizer_file_path = PathBuf::from("path/to/tokenizer.json");
    /// let special_token_map_path = PathBuf::from("path/to/special_token_map.json");
    /// let tokenizer = HFTokenizer::from_file(tokenizer_file_path, special_token_map_path)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_file<P: AsRef<Path>, S: AsRef<Path>>(
        tokenizer_file: P,
        special_token_map: S,
    ) -> Result<Self, RustBertError> {
        let tokenizer = HFBaseTokenizer::from_file(tokenizer_file)?;
        let f = File::open(&special_token_map).map_err(|e| {
            RustBertError::IOError(format!(
                "{} special token map file not found :{}",
                special_token_map.as_ref().display(),
                e
            ))
        })?;
        let br = BufReader::new(f);
        let special_token_map = serde_json::from_reader(br).map_err(|e| {
            RustBertError::IOError(format!("Invalid special token mapping file {e}"))
        })?;
        Ok(Self {
            tokenizer,
            special_token_map,
        })
    }

    fn encoding_to_tokenized_input(encoding: Encoding) -> TokenizedInput {
        let token_ids = encoding
            .get_ids()
            .iter()
            .map(|token_id| *token_id as i64)
            .collect();
        let segment_ids = encoding
            .get_type_ids()
            .iter()
            .map(|segment_id| *segment_id as i8)
            .collect();
        let special_tokens_mask = encoding
            .get_special_tokens_mask()
            .iter()
            .map(|segment_id| *segment_id as i8)
            .collect();
        let overflowing_tokens: Vec<i64> = encoding
            .get_overflowing()
            .iter()
            .flat_map(|encoding| encoding.get_ids())
            .map(|token_id| *token_id as i64)
            .collect();
        let num_truncated_tokens = overflowing_tokens.len();
        let token_offsets = encoding
            .get_offsets()
            .iter()
            .map(|offset| {
                Some(Offset {
                    begin: offset.0 as OffsetSize,
                    end: offset.1 as OffsetSize,
                })
            })
            .collect();
        let reference_offsets = encoding
            .get_offsets()
            .iter()
            .map(|offset| (offset.0 as OffsetSize..offset.1 as OffsetSize).collect())
            .collect();
        let mask = encoding
            .get_special_tokens_mask()
            .iter()
            .map(|segment_id| {
                if *segment_id == 0 {
                    Mask::None
                } else {
                    Mask::Special
                }
            })
            .collect();
        TokenizedInput {
            token_ids,
            segment_ids,
            special_tokens_mask,
            overflowing_tokens,
            num_truncated_tokens,
            token_offsets,
            reference_offsets,
            mask,
        }
    }

    /// Encode a list of texts
    ///
    /// # Arguments
    /// - `text_list` slice of string-like inputs to encode
    ///
    /// # Returns
    /// - `Vec<TokenizedInput>` containing the tokenized and encoded texts
    ///
    /// # Example
    ///
    /// ```no_run
    ///  # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::hf_tokenizers::HFTokenizer;
    /// # use std::path::PathBuf;
    /// # let tokenizer_file_path = PathBuf::from("path/to/tokenizer.json");
    /// # let special_token_map_path = PathBuf::from("path/to/special_token_map.json");
    /// let tokenizer = HFTokenizer::from_file(tokenizer_file_path, special_token_map_path)?;
    /// let texts = &["first text to encode", "second text to encode"];
    /// let output = tokenizer.encode_list(texts);
    /// # Ok(())
    /// # }
    /// ```
    pub fn encode_list<S>(&self, text_list: &[S]) -> Result<Vec<TokenizedInput>, RustBertError>
    where
        S: AsRef<str> + Sync + Send,
    {
        let encoding_inputs = text_list.iter().map(|text| text.as_ref()).collect();
        let encodings = self.tokenizer.encode_batch(encoding_inputs, true)?;
        let mut tokenized_inputs: Vec<TokenizedInput> = Vec::with_capacity(encodings.len());
        for encoding in encodings {
            tokenized_inputs.push(Self::encoding_to_tokenized_input(encoding));
        }

        Ok(tokenized_inputs)
    }

    /// Encode a list of text pairs
    ///
    /// This is used for application where the model takes 2 input sequences as an input (e.g. natural language inference).
    ///
    /// # Arguments
    /// - `text_pair_list` slice of tuples of string-like inputs to encode
    ///
    /// # Returns
    /// - `Vec<TokenizedInput>` containing the tokenized and encoded texts
    ///
    /// # Example
    ///
    /// ```no_run
    ///  # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::hf_tokenizers::HFTokenizer;
    /// # use std::path::PathBuf;
    /// # let tokenizer_file_path = PathBuf::from("path/to/tokenizer.json");
    /// # let special_token_map_path = PathBuf::from("path/to/special_token_map.json");
    /// let tokenizer = HFTokenizer::from_file(tokenizer_file_path, special_token_map_path)?;
    /// let texts = &[
    ///     (
    ///         "first text of first pair to encode",
    ///         "second text of first pair to encode",
    ///     ),
    ///     (
    ///         "first text of second pair to encode",
    ///         "second text of second pair to encode",
    ///     ),
    /// ];
    /// let output = tokenizer.encode_pair_list(texts);
    /// # Ok(())
    /// # }
    /// ```
    pub fn encode_pair_list(
        &self,
        text_pair_list: &[(&str, &str)],
    ) -> Result<Vec<TokenizedInput>, RustBertError> {
        let encoding_inputs: Vec<EncodeInput> = text_pair_list
            .iter()
            .map(|(text_1, text_2)| {
                EncodeInput::Dual(
                    InputSequence::Raw(Cow::Borrowed(text_1)),
                    InputSequence::Raw(Cow::Borrowed(text_2)),
                )
            })
            .collect();
        let encodings = self.tokenizer.encode_batch(encoding_inputs, true)?;
        let mut tokenized_inputs: Vec<TokenizedInput> = Vec::with_capacity(encodings.len());
        for encoding in encodings {
            tokenized_inputs.push(Self::encoding_to_tokenized_input(encoding));
        }

        Ok(tokenized_inputs)
    }

    /// Encode a single text pair
    ///
    /// This is used for application where the model takes 2 input sequences as an input (e.g. natural language inference).
    /// This generic method handles both the case where a second input is provided and when it is not
    /// (falling back to single sequence encoding)
    ///
    /// # Arguments
    /// - `text_1` string slice for the first text
    /// - `text_2` Optional string slice for the second text
    ///
    /// # Returns
    /// - `TokenizedInput` containing the tokenized and encoded texts
    ///
    /// # Example
    ///
    /// ```no_run
    ///  # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::hf_tokenizers::HFTokenizer;
    /// # use std::path::PathBuf;
    /// # let tokenizer_file_path = PathBuf::from("path/to/tokenizer.json");
    /// # let special_token_map_path = PathBuf::from("path/to/special_token_map.json");
    /// let tokenizer = HFTokenizer::from_file(tokenizer_file_path, special_token_map_path)?;
    /// let text_1 = "first text to encode";
    /// let output_1 = tokenizer.encode_pair(text_1, None);
    /// let text_2 = "second text to encode";
    /// let output_2 = tokenizer.encode_pair(text_1, Some(text_2));
    /// # Ok(())
    /// # }
    /// ```
    pub fn encode_pair(
        &self,
        text_1: &str,
        text_2: Option<&str>,
    ) -> Result<TokenizedInput, RustBertError> {
        let encoding_input = if let Some(text_2) = text_2 {
            EncodeInput::Dual(
                InputSequence::Raw(Cow::Borrowed(text_1)),
                InputSequence::Raw(Cow::Borrowed(text_2)),
            )
        } else {
            EncodeInput::Single(InputSequence::Raw(Cow::Borrowed(text_1)))
        };
        let encoding = self.tokenizer.encode(encoding_input, true)?;
        Ok(Self::encoding_to_tokenized_input(encoding))
    }

    /// Tokenize a text
    ///
    /// # Arguments
    /// - `text` string slice to tokenize
    ///
    /// # Returns
    /// - `Vec<String>` tokenized text
    ///
    /// # Example
    ///
    /// ```no_run
    ///  # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::hf_tokenizers::HFTokenizer;
    /// # use std::path::PathBuf;
    /// # let tokenizer_file_path = PathBuf::from("path/to/tokenizer.json");
    /// # let special_token_map_path = PathBuf::from("path/to/special_token_map.json");
    /// let tokenizer = HFTokenizer::from_file(tokenizer_file_path, special_token_map_path)?;
    /// let text = "first text to encode";
    /// let output = tokenizer.tokenize(text);
    /// # Ok(())
    /// # }
    /// ```
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.tokenizer
            .encode(text, false)
            .unwrap()
            .get_tokens()
            .to_vec()
    }

    /// Tokenize a list of texts
    ///
    /// # Arguments
    /// - `texts` slice of string-like references to tokenize
    ///
    /// # Returns
    /// - `Vec<Vec<String>>` tokenized texts
    ///
    /// # Example
    ///
    /// ```no_run
    ///  # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::hf_tokenizers::HFTokenizer;
    /// # use std::path::PathBuf;
    /// # let tokenizer_file_path = PathBuf::from("path/to/tokenizer.json");
    /// # let special_token_map_path = PathBuf::from("path/to/special_token_map.json");
    /// let tokenizer = HFTokenizer::from_file(tokenizer_file_path, special_token_map_path)?;
    /// let texts = &["first text to encode", "second text to encode"];
    /// let output = tokenizer.tokenize_list(texts);
    /// # Ok(())
    /// # }
    /// ```
    pub fn tokenize_list<S>(&self, texts: &[S]) -> Vec<Vec<String>>
    where
        S: AsRef<str> + Send + Sync,
    {
        texts
            .iter()
            .map(|text| self.tokenize(text.as_ref()))
            .collect()
    }

    /// Tokenize a text with offsets information
    ///
    /// # Arguments
    /// - `text` string slice to tokenize with offsets
    ///
    /// # Returns
    /// - `Vec<String>` tokenized text
    ///
    /// # Example
    ///
    /// ```no_run
    ///  # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::hf_tokenizers::HFTokenizer;
    /// # use std::path::PathBuf;
    /// # let tokenizer_file_path = PathBuf::from("path/to/tokenizer.json");
    /// # let special_token_map_path = PathBuf::from("path/to/special_token_map.json");
    /// let tokenizer = HFTokenizer::from_file(tokenizer_file_path, special_token_map_path)?;
    /// let text = "first text to encode";
    /// let output = tokenizer.tokenize_with_offsets(text);
    /// # Ok(())
    /// # }
    /// ```
    pub fn tokenize_with_offsets(&self, text: &str) -> TokensWithOffsets {
        let encoding = self.tokenizer.encode(text, false).unwrap();
        let tokens = encoding.get_tokens().to_vec();
        let offsets = encoding
            .get_offsets()
            .iter()
            .map(|offset| {
                Some(Offset {
                    begin: offset.0 as OffsetSize,
                    end: offset.1 as OffsetSize,
                })
            })
            .collect();
        let reference_offsets = encoding
            .get_offsets()
            .iter()
            .map(|offset| (offset.0 as OffsetSize..offset.1 as OffsetSize).collect())
            .collect();
        let masks = encoding
            .get_special_tokens_mask()
            .iter()
            .map(|segment_id| {
                if *segment_id == 0 {
                    Mask::None
                } else {
                    Mask::Special
                }
            })
            .collect();
        TokensWithOffsets {
            tokens,
            offsets,
            reference_offsets,
            masks,
        }
    }

    /// Decode a sequence of token id to a text
    ///
    /// # Arguments
    /// - `token_ids` slice of token ids
    ///- `skip_special_token_ids` flag indicating if special token ids should be skipped during decoding
    ///
    /// # Returns
    /// - `String` decoded text
    ///
    /// # Example
    ///
    /// ```no_run
    ///  # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::hf_tokenizers::HFTokenizer;
    /// # use std::path::PathBuf;
    /// # let tokenizer_file_path = PathBuf::from("path/to/tokenizer.json");
    /// # let special_token_map_path = PathBuf::from("path/to/special_token_map.json");
    /// let tokenizer = HFTokenizer::from_file(tokenizer_file_path, special_token_map_path)?;
    /// let token_ids = &[0, 2, 5, 9, 4, 2, 1];
    /// let skip_special_token_ids = true;
    /// let output = tokenizer.decode(token_ids, skip_special_token_ids);
    /// # Ok(())
    /// # }
    /// ```
    pub fn decode(&self, token_ids: &[i64], skip_special_tokens: bool) -> String {
        self.tokenizer
            .decode(
                token_ids
                    .iter()
                    .map(|token_id| *token_id as u32)
                    .collect::<Vec<u32>>()
                    .as_slice(),
                skip_special_tokens,
            )
            .unwrap()
    }

    fn token_ids_with_offsets_to_encoding(
        &self,
        token_ids_with_offsets: TokenIdsWithOffsets,
    ) -> Encoding {
        let ids: Vec<u32> = token_ids_with_offsets
            .ids
            .iter()
            .map(|token_id| *token_id as u32)
            .collect();
        let type_ids = token_ids_with_offsets
            .ids
            .iter()
            .map(|segment_id| *segment_id as u32)
            .collect();
        let tokens = ids
            .iter()
            .map(|token_id| {
                self.tokenizer
                    .id_to_token(*token_id)
                    .unwrap_or(self.tokenizer.decode(&[*token_id], false).unwrap())
            })
            .collect();
        let words = vec![None::<u32>; ids.len()];
        let offsets = token_ids_with_offsets
            .offsets
            .iter()
            .map(|offset| {
                offset
                    .map(|offset| (offset.begin as usize, offset.end as usize))
                    .unwrap_or((0, 0))
            })
            .collect();
        let special_tokens_mask = token_ids_with_offsets
            .masks
            .iter()
            .map(|segment_id| match segment_id {
                Mask::Special => 1,
                _ => 0,
            })
            .collect();
        let overflowing: Vec<Encoding> = vec![];
        let attention_mask = vec![1; ids.len()];
        let sequence_ranges = HashMap::new();
        Encoding::new(
            ids,
            type_ids,
            tokens,
            words,
            offsets,
            special_tokens_mask,
            attention_mask,
            overflowing,
            sequence_ranges,
        )
    }

    /// Post-process a sequence or sequence pair
    ///
    /// Adds the special token for single/pair of sequences and apply tokenizer post-processing
    ///
    /// # Arguments
    /// - `token_ids_with_offsets_1` first sequence's `TokenIdsWithOffsets`
    /// - `token_ids_with_offsets_2` optional second sequence's `TokenIdsWithOffsets`
    ///
    /// # Returns
    /// - `TokenizedInput` psot-processed encoding for the inputs provided.
    ///
    /// # Example
    ///
    /// ```no_run
    ///  # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::hf_tokenizers::HFTokenizer;
    /// # use std::path::PathBuf;
    /// use rust_tokenizers::{Offset, TokenIdsWithOffsets};
    /// # let tokenizer_file_path = PathBuf::from("path/to/tokenizer.json");
    /// # let special_token_map_path = PathBuf::from("path/to/special_token_map.json");
    /// let tokenizer = HFTokenizer::from_file(tokenizer_file_path, special_token_map_path)?;
    /// let token_ids_with_offsets_1 = TokenIdsWithOffsets {
    ///     ids: vec![0, 1, 2],
    ///     offsets: vec![
    ///         Some(Offset { begin: 0, end: 1 }),
    ///         Some(Offset { begin: 1, end: 2 }),
    ///         Some(Offset { begin: 2, end: 3 }),
    ///     ],
    ///     reference_offsets: vec![vec![0], vec![1], vec![2]],
    ///     masks: vec![],
    /// };
    /// let token_ids_with_offsets_2 = TokenIdsWithOffsets {
    ///     ids: vec![8, 9, 10],
    ///     offsets: vec![
    ///         Some(Offset { begin: 3, end: 4 }),
    ///         Some(Offset { begin: 4, end: 5 }),
    ///         Some(Offset { begin: 5, end: 6 }),
    ///     ],
    ///     reference_offsets: vec![vec![3], vec![4], vec![5]],
    ///     masks: vec![],
    /// };
    /// let output = tokenizer
    ///     .build_input_with_special_tokens(token_ids_with_offsets_1, Some(token_ids_with_offsets_2));
    /// # Ok(())
    /// # }
    /// ```
    pub fn build_input_with_special_tokens(
        &self,
        token_ids_with_offsets_1: TokenIdsWithOffsets,
        token_ids_with_offsets_2: Option<TokenIdsWithOffsets>,
    ) -> TokenizedInput {
        let encoding_1 = self.token_ids_with_offsets_to_encoding(token_ids_with_offsets_1);
        let encoding_2 = token_ids_with_offsets_2
            .map(|encoding| self.token_ids_with_offsets_to_encoding(encoding));
        let encoding_output = self
            .tokenizer
            .post_process(encoding_1, encoding_2, true)
            .unwrap();
        Self::encoding_to_tokenized_input(encoding_output)
    }

    /// Converts a single token to a token id
    ///
    /// Returns the unknown token id if the item is not present in the tokenizer vocabulary.
    ///
    /// # Arguments
    /// - `token` string slice to convert
    ///
    /// # Returns
    /// - `i64` token id (or unknown token id if not found in the vocabulary)
    ///
    /// # Example
    ///
    /// ```no_run
    ///  # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::hf_tokenizers::HFTokenizer;
    /// # use std::path::PathBuf;
    /// use rust_tokenizers::{Offset, TokenIdsWithOffsets};
    /// # let tokenizer_file_path = PathBuf::from("path/to/tokenizer.json");
    /// # let special_token_map_path = PathBuf::from("path/to/special_token_map.json");
    /// let tokenizer = HFTokenizer::from_file(tokenizer_file_path, special_token_map_path)?;
    /// let token = "Hello";
    /// let output = tokenizer.token_to_id(token);
    /// # Ok(())
    /// # }
    /// ```
    pub fn token_to_id(&self, token: &str) -> i64 {
        self.tokenizer.token_to_id(token.as_ref()).unwrap_or(
            self.tokenizer
                .token_to_id(self.special_token_map.unk_token.as_str())
                .unwrap(),
        ) as i64
    }

    /// Converts a slice of tokens to  token ids
    ///
    /// Returns the unknown token id if the item is not present in the tokenizer vocabulary.
    ///
    /// # Arguments
    /// - `tokens` slice of string slices to convert
    ///
    /// # Returns
    /// - `Vec<i64>` token ids (with unknown token id at position of items not found in the vocabulary)
    ///
    /// # Example
    ///
    /// ```no_run
    ///  # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::hf_tokenizers::HFTokenizer;
    /// # use std::path::PathBuf;
    /// use rust_tokenizers::{Offset, TokenIdsWithOffsets};
    /// # let tokenizer_file_path = PathBuf::from("path/to/tokenizer.json");
    /// # let special_token_map_path = PathBuf::from("path/to/special_token_map.json");
    /// let tokenizer = HFTokenizer::from_file(tokenizer_file_path, special_token_map_path)?;
    /// let tokens = &["Hello", "world", "!"];
    /// let output = tokenizer.convert_tokens_to_ids(tokens);
    /// # Ok(())
    /// # }
    /// ```
    pub fn convert_tokens_to_ids<S>(&self, tokens: &[S]) -> Vec<i64>
    where
        S: AsRef<str>,
    {
        tokens
            .iter()
            .map(|token| self.token_to_id(token.as_ref()))
            .collect()
    }

    /// Add tokens to the tokenizer vocabulary
    ///
    /// These tokens are not used by the tokenization algorithm and simply added to the vocabulary
    ///
    /// # Arguments
    /// - `tokens` tokens to add to the vocabulary
    ///
    /// # Example
    ///
    /// ```no_run
    ///  # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::hf_tokenizers::HFTokenizer;
    /// # use std::path::PathBuf;
    /// use rust_tokenizers::{Offset, TokenIdsWithOffsets};
    /// # let tokenizer_file_path = PathBuf::from("path/to/tokenizer.json");
    /// # let special_token_map_path = PathBuf::from("path/to/special_token_map.json");
    /// let mut tokenizer = HFTokenizer::from_file(tokenizer_file_path, special_token_map_path)?;
    /// tokenizer.add_tokens(&["<CLS>", "<SEP>"]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_tokens(&mut self, tokens: &[&str]) {
        let added_tokens = tokens
            .iter()
            .map(|token| AddedToken {
                content: token.to_string(),
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: false,
                special: false,
            })
            .collect::<Vec<AddedToken>>();
        self.tokenizer.add_tokens(&added_tokens);
    }

    /// Add extra token ids to the tokenizer vocabulary
    ///
    /// These tokens are automatically formatted as "<extra_id_{extra_id}>"
    ///
    /// # Arguments
    /// - `num_extra_ids` number of tokens to add
    ///
    /// # Example
    ///
    /// ```no_run
    ///  # fn main() -> anyhow::Result<()> {
    /// # use rust_bert::pipelines::hf_tokenizers::HFTokenizer;
    /// # use std::path::PathBuf;
    /// use rust_tokenizers::{Offset, TokenIdsWithOffsets};
    /// # let tokenizer_file_path = PathBuf::from("path/to/tokenizer.json");
    /// # let special_token_map_path = PathBuf::from("path/to/special_token_map.json");
    /// let mut tokenizer = HFTokenizer::from_file(tokenizer_file_path, special_token_map_path)?;
    /// tokenizer.add_extra_ids(42);
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_extra_ids(&mut self, num_extra_ids: i64) {
        let mut added_tokens: Vec<AddedToken> = Vec::with_capacity(num_extra_ids as usize);
        for extra_id in 0..num_extra_ids {
            added_tokens.push(AddedToken {
                content: format!("<extra_id_{extra_id}>"),
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: false,
                special: false,
            });
        }
        self.tokenizer.add_tokens(&added_tokens);
    }
}
