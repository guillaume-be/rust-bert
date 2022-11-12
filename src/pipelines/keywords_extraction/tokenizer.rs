use crate::pipelines::keywords_extraction::stopwords::ENGLISH_STOPWORDS;
use regex::Regex;
use rust_tokenizers::{Offset, OffsetSize};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

const DEFAULT_REGEX_PATTERN: &str = r"(?u)\b\w\w+\b";

pub struct StopWordsTokenizer<'a> {
    stopwords: HashSet<&'a str>,
    pattern: Regex,
    do_lower_case: bool,
}

impl<'a> StopWordsTokenizer<'a> {
    pub fn new(
        stopwords: Option<HashSet<&'a str>>,
        pattern: Option<Regex>,
        do_lower_case: bool,
    ) -> Self {
        let stopwords = stopwords.unwrap_or_else(|| HashSet::from(ENGLISH_STOPWORDS));
        let pattern = pattern.unwrap_or_else(|| Regex::new(DEFAULT_REGEX_PATTERN).unwrap());

        Self {
            stopwords,
            pattern,
            do_lower_case,
        }
    }

    pub fn tokenize<'b>(
        &self,
        text: &'b str,
        ngram_range: (usize, usize),
    ) -> HashMap<Cow<'b, str>, Vec<Offset>> {
        let mut tokenized_text = HashMap::new();

        let mut tokens_list = Vec::new();
        for hit in self.pattern.find_iter(text) {
            let pos = Offset {
                begin: hit.start() as OffsetSize,
                end: hit.end() as OffsetSize,
            };
            tokens_list.push(pos);
        }
        for ngram_size in ngram_range.0..ngram_range.1 + 1 {
            'ngram_loop: for ngram in tokens_list.windows(ngram_size) {
                let pos = Offset {
                    begin: ngram[0].begin,
                    end: ngram.last().unwrap().end,
                };
                let mut ngram_text = Cow::from(&text[pos.begin as usize..pos.end as usize]);
                if self.do_lower_case {
                    ngram_text = Cow::from(ngram_text.to_lowercase());
                }
                if self.stopwords.contains(&*ngram_text) {
                    continue;
                }
                if ngram_size > 1 {
                    for token in ngram {
                        let mut token = Cow::from(&text[token.begin as usize..token.end as usize]);
                        if self.do_lower_case {
                            token = Cow::from(token.to_lowercase());
                        }
                        if self.stopwords.contains(&*token) {
                            continue 'ngram_loop;
                        }
                    }
                    if ngram.last().unwrap().begin > ngram[0].end + 1 {
                        continue;
                    }
                }
                tokenized_text
                    .entry(ngram_text)
                    .and_modify(|pos_vec: &mut Vec<Offset>| pos_vec.push(pos))
                    .or_insert_with(|| vec![pos]);
            }
        }
        tokenized_text
    }

    pub fn tokenize_list<'b, S>(
        &self,
        texts: &'b [S],
        ngram_range: (usize, usize),
    ) -> Vec<HashMap<Cow<'b, str>, Vec<Offset>>>
    where
        S: AsRef<str> + Sync,
    {
        texts
            .iter()
            .map(|text| self.tokenize(text.as_ref(), ngram_range))
            .collect()
    }
}
