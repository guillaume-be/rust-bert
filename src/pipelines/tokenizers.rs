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

#[derive(Debug, Default, Clone, Deserialize)]
pub struct SpecialTokenMap {
    pub unk_token: String,
    #[serde(default)]
    #[serde(deserialize_with = "string_or_added_token_struct")]
    pub pad_token: Option<String>,
    #[serde(default)]
    #[serde(deserialize_with = "string_or_added_token_struct")]
    pub bos_token: Option<String>,
    #[serde(default)]
    #[serde(deserialize_with = "string_or_added_token_struct")]
    pub sep_token: Option<String>,
    #[serde(default)]
    #[serde(deserialize_with = "string_or_added_token_struct")]
    pub cls_token: Option<String>,
    #[serde(default)]
    #[serde(deserialize_with = "string_or_added_token_struct")]
    pub eos_token: Option<String>,
    #[serde(default)]
    #[serde(deserialize_with = "string_or_added_token_struct")]
    pub mask_token: Option<String>,
    pub additional_special_tokens: Option<HashSet<String>>,
}

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

    Ok(deserializer.deserialize_any(StringOrStruct)?)
}

pub struct HFTokenizer {
    tokenizer: HFBaseTokenizer,
    pub(crate) special_token_map: SpecialTokenMap,
}

impl HFTokenizer {
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
            .into_iter()
            .map(|token_id| *token_id as i64)
            .collect();
        let segment_ids = encoding
            .get_type_ids()
            .into_iter()
            .map(|segment_id| *segment_id as i8)
            .collect();
        let special_tokens_mask = encoding
            .get_special_tokens_mask()
            .into_iter()
            .map(|segment_id| *segment_id as i8)
            .collect();
        let overflowing_tokens: Vec<i64> = encoding
            .get_overflowing()
            .iter()
            .map(|encoding| encoding.get_ids())
            .flatten()
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
            .into_iter()
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

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.tokenizer
            .encode(text, false)
            .unwrap()
            .get_tokens()
            .to_vec()
    }

    pub fn tokenize_list<S>(&self, text: &[S]) -> Vec<Vec<String>>
    where
        S: AsRef<str> + Send + Sync,
    {
        text.iter()
            .map(|text| self.tokenize(text.as_ref()))
            .collect()
    }

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
            .into_iter()
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

    pub fn decode(&self, token_ids: &[i64], skip_special_tokens: bool) -> String {
        self.tokenizer
            .decode(
                token_ids.iter().map(|token_id| *token_id as u32).collect(),
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
                    .unwrap_or(self.tokenizer.decode(vec![*token_id], false).unwrap())
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

    pub fn token_to_id(&self, token: &str) -> i64 {
        self.tokenizer.token_to_id(token.as_ref()).unwrap_or(
            self.tokenizer
                .token_to_id(self.special_token_map.unk_token.as_str())
                .unwrap(),
        ) as i64
    }

    pub fn convert_tokens_to_ids<S>(&self, tokens: &[S]) -> Vec<i64>
    where
        S: AsRef<str>,
    {
        tokens
            .iter()
            .map(|token| self.token_to_id(token.as_ref()))
            .collect()
    }

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
