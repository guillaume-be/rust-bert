use crate::pipelines::keywords_extraction::stopwords::ENGLISH_STOPWORDS;
use regex::Regex;
use std::collections::{HashMap, HashSet};

const DEFAULT_REGEX_PATTERN: &str = r"(?u)\b\w\w+\b";

pub struct StopWordsTokenizer<'a> {
    stopwords: HashSet<&'a str>,
    pattern: Regex,
}

impl<'a> StopWordsTokenizer<'a> {
    pub fn new(stopwords: Option<HashSet<&'a str>>, pattern: Option<Regex>) -> Self {
        let stopwords = stopwords.unwrap_or_else(|| HashSet::from(ENGLISH_STOPWORDS));
        let pattern = pattern.unwrap_or_else(|| Regex::new(DEFAULT_REGEX_PATTERN).unwrap());

        Self { stopwords, pattern }
    }

    pub fn tokenize<'b>(&self, text: &'b str) -> HashMap<&'b str, Vec<(usize, usize)>> {
        let mut tokenized_text = HashMap::new();

        for hit in self.pattern.find_iter(text) {
            let hit_text = hit.as_str().to_lowercase();
            if self.stopwords.contains(hit_text.as_str()) {
                continue;
            }
            let pos = (hit.start(), hit.end());
            tokenized_text
                .entry(&text[pos.0..pos.1])
                .and_modify(|pos_vec: &mut Vec<(usize, usize)>| pos_vec.push(pos))
                .or_insert(vec![pos]);
        }
        tokenized_text
    }

    pub fn tokenize_list<'b, S>(&self, texts: &'b [S]) -> Vec<HashMap<&'b str, Vec<(usize, usize)>>>
    where
        S: AsRef<str> + Sync,
    {
        texts
            .into_iter()
            .map(|text| self.tokenize(text.as_ref()))
            .collect()
    }
}
