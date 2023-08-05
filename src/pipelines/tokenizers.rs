use crate::RustBertError;
use rust_tokenizers::{Mask, Offset, OffsetSize, TokenizedInput};
use serde::{de, Deserialize, Deserializer};
use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use tokenizers::tokenizer::Tokenizer as HFBaseTokenizer;
use tokenizers::Encoding;

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
    special_token_map: SpecialTokenMap,
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
        S: AsRef<str> + Sync + Send + Clone,
    {
        let encoding_inputs = text_list.iter().map(|text| text.as_ref()).collect();
        let mut encodings = self.tokenizer.encode_batch(encoding_inputs, true)?;
        let mut tokenized_inputs: Vec<TokenizedInput> = Vec::with_capacity(encodings.len());
        for encoding in encodings {
            tokenized_inputs.push(Self::encoding_to_tokenized_input(encoding));
        }

        Ok(tokenized_inputs)
    }
}
