/// # Pretrained config files for sentence embeddings
pub struct SentenceEmbeddingsModulesConfigResources;

/// # Pretrained dense weights files for sentence embeddings
pub struct SentenceEmbeddingsDenseResources;

/// # Pretrained dense config files for sentence embeddings
pub struct SentenceEmbeddingsDenseConfigResources;

/// # Pretrained pooling config files for sentence embeddings
pub struct SentenceEmbeddingsPoolingConfigResources;

/// # Pretrained config files for sentence embeddings
pub struct SentenceEmbeddingsConfigResources;

/// # Pretrained tokenizer config files for sentence embeddings
pub struct SentenceEmbeddingsTokenizerConfigResources;

pub enum SentenceEmbeddingsModelType {
    DistiluseBaseMultilingualCased,
    BertBaseNliMeanTokens,
    AllMiniLmL12V2,
    AllMiniLmL6V2,
    AllDistilrobertaV1,
    ParaphraseAlbertSmallV2,
    SentenceT5Base,
}

impl SentenceEmbeddingsModulesConfigResources {
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased>. Modified with conversion to C-array format.
    pub const DISTILUSE_BASE_MULTILINGUAL_CASED: (&'static str, &'static str) = (
        "distiluse-base-multilingual-cased/sbert-config",
        "https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased/resolve/main/modules.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens>. Modified with conversion to C-array format.
    pub const BERT_BASE_NLI_MEAN_TOKENS: (&'static str, &'static str) = (
        "bert-base-nli-mean-tokens/sbert-config",
        "https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens/resolve/main/modules.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L12_V2: (&'static str, &'static str) = (
        "all-mini-lm-l12-v2/sbert-config",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/modules.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L6_V2: (&'static str, &'static str) = (
        "all-mini-lm-l6-v2/sbert-config",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json",
    );
    /// Shared under Apache 2.0 licenseat <https://huggingface.co/sentence-transformers/all-distilroberta-v1>. Modified with conversion to C-array format.
    pub const ALL_DISTILROBERTA_V1: (&'static str, &'static str) = (
        "all-distilroberta-v1/sbert-config",
        "https://huggingface.co/sentence-transformers/all-distilroberta-v1/resolve/main/modules.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2>. Modified with conversion to C-array format.
    pub const PARAPHRASE_ALBERT_SMALL_V2: (&'static str, &'static str) = (
        "paraphrase-albert-small-v2/sbert-config",
        "https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2/resolve/main/modules.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/sentence-t5-base>. Modified with conversion to C-array format.
    pub const SENTENCE_T5_BASE: (&'static str, &'static str) = (
        "sentence-t5-base/sbert-config",
        "https://huggingface.co/sentence-transformers/sentence-t5-base/resolve/main/modules.json",
    );
}

impl SentenceEmbeddingsDenseResources {
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased>. Modified with conversion to C-array format.
    pub const DISTILUSE_BASE_MULTILINGUAL_CASED: (&'static str, &'static str) = (
        "distiluse-base-multilingual-cased/sbert-dense",
        "https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased/resolve/main/2_Dense/rust_model.ot",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/sentence-t5-base>. Modified with conversion to C-array format.
    pub const SENTENCE_T5_BASE: (&'static str, &'static str) = (
        "sentence-t5-base/sbert-dense",
        "https://huggingface.co/sentence-transformers/sentence-t5-base/resolve/main/2_Dense/rust_model.ot",
    );
}

impl SentenceEmbeddingsDenseConfigResources {
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased>. Modified with conversion to C-array format.
    pub const DISTILUSE_BASE_MULTILINGUAL_CASED: (&'static str, &'static str) = (
        "distiluse-base-multilingual-cased/sbert-dense-config",
        "https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased/resolve/main/2_Dense/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/sentence-t5-base>. Modified with conversion to C-array format.
    pub const SENTENCE_T5_BASE: (&'static str, &'static str) = (
        "sentence-t5-base/sbert-dense-config",
        "https://huggingface.co/sentence-transformers/sentence-t5-base/resolve/main/2_Dense/config.json",
    );
}

impl SentenceEmbeddingsPoolingConfigResources {
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased>. Modified with conversion to C-array format.
    pub const DISTILUSE_BASE_MULTILINGUAL_CASED: (&'static str, &'static str) = (
        "distiluse-base-multilingual-cased/sbert-pooling-config",
        "https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased/resolve/main/1_Pooling/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens>. Modified with conversion to C-array format.
    pub const BERT_BASE_NLI_MEAN_TOKENS: (&'static str, &'static str) = (
        "bert-base-nli-mean-tokens/sbert-pooling-config",
        "https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens/resolve/main/1_Pooling/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L12_V2: (&'static str, &'static str) = (
        "all-mini-lm-l12-v2/sbert-pooling-config",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/1_Pooling/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L6_V2: (&'static str, &'static str) = (
        "all-mini-lm-l6-v2/sbert-pooling-config",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/1_Pooling/config.json",
    );
    /// Shared under Apache 2.0 licenseat <https://huggingface.co/sentence-transformers/all-distilroberta-v1>. Modified with conversion to C-array format.
    pub const ALL_DISTILROBERTA_V1: (&'static str, &'static str) = (
        "all-distilroberta-v1/sbert-pooling-config",
        "https://huggingface.co/sentence-transformers/all-distilroberta-v1/resolve/main/1_Pooling/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2>. Modified with conversion to C-array format.
    pub const PARAPHRASE_ALBERT_SMALL_V2: (&'static str, &'static str) = (
        "paraphrase-albert-small-v2/sbert-pooling-config",
        "https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2/resolve/main/1_Pooling/config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/sentence-t5-base>. Modified with conversion to C-array format.
    pub const SENTENCE_T5_BASE: (&'static str, &'static str) = (
        "sentence-t5-base/sbert-pooling-config",
        "https://huggingface.co/sentence-transformers/sentence-t5-base/resolve/main/1_Pooling/config.json",
    );
}

impl SentenceEmbeddingsConfigResources {
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased>. Modified with conversion to C-array format.
    pub const DISTILUSE_BASE_MULTILINGUAL_CASED: (&'static str, &'static str) = (
        "distiluse-base-multilingual-cased/sbert-config",
        "https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased/resolve/main/sentence_bert_config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens>. Modified with conversion to C-array format.
    pub const BERT_BASE_NLI_MEAN_TOKENS: (&'static str, &'static str) = (
        "bert-base-nli-mean-tokens/sbert-config",
        "https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens/resolve/main/sentence_bert_config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L12_V2: (&'static str, &'static str) = (
        "all-mini-lm-l12-v2/sbert-config",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/sentence_bert_config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L6_V2: (&'static str, &'static str) = (
        "all-mini-lm-l6-v2/sbert-config",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/sentence_bert_config.json",
    );
    /// Shared under Apache 2.0 licenseat <https://huggingface.co/sentence-transformers/all-distilroberta-v1>. Modified with conversion to C-array format.
    pub const ALL_DISTILROBERTA_V1: (&'static str, &'static str) = (
        "all-distilroberta-v1/sbert-config",
        "https://huggingface.co/sentence-transformers/all-distilroberta-v1/resolve/main/sentence_bert_config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2>. Modified with conversion to C-array format.
    pub const PARAPHRASE_ALBERT_SMALL_V2: (&'static str, &'static str) = (
        "paraphrase-albert-small-v2/sbert-config",
        "https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2/resolve/main/sentence_bert_config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/sentence-t5-base>. Modified with conversion to C-array format.
    pub const SENTENCE_T5_BASE: (&'static str, &'static str) = (
        "sentence-t5-base/sbert-config",
        "https://huggingface.co/sentence-transformers/sentence-t5-base/resolve/main/sentence_bert_config.json",
    );
}

impl SentenceEmbeddingsTokenizerConfigResources {
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased>. Modified with conversion to C-array format.
    pub const DISTILUSE_BASE_MULTILINGUAL_CASED: (&'static str, &'static str) = (
        "distiluse-base-multilingual-cased/tokenizer-config",
        "https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased/resolve/main/tokenizer_config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens>. Modified with conversion to C-array format.
    pub const BERT_BASE_NLI_MEAN_TOKENS: (&'static str, &'static str) = (
        "bert-base-nli-mean-tokens/tokenizer-config",
        "https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens/resolve/main/tokenizer_config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L12_V2: (&'static str, &'static str) = (
        "all-mini-lm-l12-v2/tokenizer-config",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/tokenizer_config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>. Modified with conversion to C-array format.
    pub const ALL_MINI_LM_L6_V2: (&'static str, &'static str) = (
        "all-mini-lm-l6-v2/tokenizer-config",
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json",
    );
    /// Shared under Apache 2.0 licenseat <https://huggingface.co/sentence-transformers/all-distilroberta-v1>. Modified with conversion to C-array format.
    pub const ALL_DISTILROBERTA_V1: (&'static str, &'static str) = (
        "all-distilroberta-v1/tokenizer-config",
        "https://huggingface.co/sentence-transformers/all-distilroberta-v1/resolve/main/tokenizer_config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2>. Modified with conversion to C-array format.
    pub const PARAPHRASE_ALBERT_SMALL_V2: (&'static str, &'static str) = (
        "paraphrase-albert-small-v2/tokenizer-config",
        "https://huggingface.co/sentence-transformers/paraphrase-albert-small-v2/resolve/main/tokenizer_config.json",
    );
    /// Shared under Apache 2.0 license at <https://huggingface.co/sentence-transformers/sentence-t5-base>. Modified with conversion to C-array format.
    pub const SENTENCE_T5_BASE: (&'static str, &'static str) = (
        "sentence-t5-base/tokenizer-config",
        "https://huggingface.co/sentence-transformers/sentence-t5-base/resolve/main/tokenizer_config.json",
    );
}
