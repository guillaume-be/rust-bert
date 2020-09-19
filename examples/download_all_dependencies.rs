extern crate anyhow;

use rust_bert::albert::{AlbertConfigResources, AlbertModelResources, AlbertVocabResources};
use rust_bert::bart::{
    BartConfigResources, BartMergesResources, BartModelResources, BartVocabResources,
};
use rust_bert::bert::{BertConfigResources, BertModelResources, BertVocabResources};
use rust_bert::distilbert::{
    DistilBertConfigResources, DistilBertModelResources, DistilBertVocabResources,
};
use rust_bert::electra::{ElectraConfigResources, ElectraModelResources, ElectraVocabResources};
use rust_bert::gpt2::{
    Gpt2ConfigResources, Gpt2MergesResources, Gpt2ModelResources, Gpt2VocabResources,
};
use rust_bert::openai_gpt::{
    OpenAiGptConfigResources, OpenAiGptMergesResources, OpenAiGptModelResources,
    OpenAiGptVocabResources,
};
use rust_bert::resources::{RemoteResource, Resource};
use rust_bert::roberta::{
    RobertaConfigResources, RobertaMergesResources, RobertaModelResources, RobertaVocabResources,
};
use rust_bert::t5::{T5ConfigResources, T5ModelResources, T5VocabResources};
use rust_bert::xlnet::{XLNetConfigResources, XLNetModelResources, XLNetVocabResources};

/// This example downloads and caches all dependencies used in model tests. This allows for safe
/// multi threaded testing (two test using the same resource would otherwise download the file to
/// the same location).

fn download_distil_gpt2() -> anyhow::Result<()> {
    //   Shared under Apache 2.0 license by the HuggingFace Inc. team at https://huggingface.co/models. Modified with conversion to C-array format.
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        Gpt2ConfigResources::DISTIL_GPT2,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        Gpt2VocabResources::DISTIL_GPT2,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        Gpt2MergesResources::DISTIL_GPT2,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        Gpt2ModelResources::DISTIL_GPT2,
    ));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = merges_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_distilbert_sst2() -> anyhow::Result<()> {
    //   Shared under Apache 2.0 license by the HuggingFace Inc. team at https://huggingface.co/models. Modified with conversion to C-array format.
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        DistilBertModelResources::DISTIL_BERT_SST2,
    ));
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        DistilBertConfigResources::DISTIL_BERT_SST2,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        DistilBertVocabResources::DISTIL_BERT_SST2,
    ));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_distilbert_qa() -> anyhow::Result<()> {
    //   Shared under Apache 2.0 license by the HuggingFace Inc. team at https://huggingface.co/models. Modified with conversion to C-array format.
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        DistilBertModelResources::DISTIL_BERT_SQUAD,
    ));
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        DistilBertConfigResources::DISTIL_BERT_SQUAD,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        DistilBertVocabResources::DISTIL_BERT_SQUAD,
    ));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_distilbert() -> anyhow::Result<()> {
    //   Shared under Apache 2.0 license by the HuggingFace Inc. team at https://huggingface.co/models. Modified with conversion to C-array format.
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        DistilBertModelResources::DISTIL_BERT,
    ));
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        DistilBertConfigResources::DISTIL_BERT,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        DistilBertVocabResources::DISTIL_BERT,
    ));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_gpt2() -> anyhow::Result<()> {
    //   Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2. Modified with conversion to C-array format.
    let config_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let weights_resource =
        Resource::Remote(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = merges_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_gpt() -> anyhow::Result<()> {
    //   Shared under MIT license by the OpenAI team at https://github.com/openai/finetune-transformer-lm. Modified with conversion to C-array format.
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        OpenAiGptConfigResources::GPT,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        OpenAiGptVocabResources::GPT,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        OpenAiGptMergesResources::GPT,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        OpenAiGptModelResources::GPT,
    ));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = merges_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_roberta() -> anyhow::Result<()> {
    //   Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaConfigResources::ROBERTA,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaVocabResources::ROBERTA,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaMergesResources::ROBERTA,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaModelResources::ROBERTA,
    ));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = merges_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_bert() -> anyhow::Result<()> {
    //   Shared under Apache 2.0 license by the Google team at https://github.com/google-research/bert. Modified with conversion to C-array format.
    let config_resource =
        Resource::Remote(RemoteResource::from_pretrained(BertConfigResources::BERT));
    let vocab_resource =
        Resource::Remote(RemoteResource::from_pretrained(BertVocabResources::BERT));
    let weights_resource =
        Resource::Remote(RemoteResource::from_pretrained(BertModelResources::BERT));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_bert_ner() -> anyhow::Result<()> {
    //    Shared under MIT license by the MDZ Digital Library team at the Bavarian State Library at https://github.com/dbmdz/berts. Modified with conversion to C-array format.
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        BertConfigResources::BERT_NER,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        BertVocabResources::BERT_NER,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        BertModelResources::BERT_NER,
    ));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_bart() -> anyhow::Result<()> {
    //   Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    let config_resource =
        Resource::Remote(RemoteResource::from_pretrained(BartConfigResources::BART));
    let vocab_resource =
        Resource::Remote(RemoteResource::from_pretrained(BartVocabResources::BART));
    let merges_resource =
        Resource::Remote(RemoteResource::from_pretrained(BartMergesResources::BART));
    let weights_resource =
        Resource::Remote(RemoteResource::from_pretrained(BartModelResources::BART));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = merges_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_bart_cnn() -> anyhow::Result<()> {
    //   Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq. Modified with conversion to C-array format.
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        BartConfigResources::BART_CNN,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        BartVocabResources::BART_CNN,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        BartMergesResources::BART_CNN,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        BartModelResources::BART_CNN,
    ));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = merges_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_electra_generator() -> anyhow::Result<()> {
    //  Shared under Apache 2.0 license by the Google team at https://github.com/google-research/electra. Modified with conversion to C-array format.
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        ElectraConfigResources::BASE_GENERATOR,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        ElectraVocabResources::BASE_GENERATOR,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        ElectraModelResources::BASE_GENERATOR,
    ));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_electra_discriminator() -> anyhow::Result<()> {
    //  Shared under Apache 2.0 license by the Google team at https://github.com/google-research/electra. Modified with conversion to C-array format.
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        ElectraConfigResources::BASE_DISCRIMINATOR,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        ElectraVocabResources::BASE_DISCRIMINATOR,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        ElectraModelResources::BASE_DISCRIMINATOR,
    ));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_albert_base_v2() -> anyhow::Result<()> {
    // Shared under Apache 2.0 license by the Google team at https://github.com/google-research/ALBERT. Modified with conversion to C-array format.
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        AlbertConfigResources::ALBERT_BASE_V2,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        AlbertVocabResources::ALBERT_BASE_V2,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        AlbertModelResources::ALBERT_BASE_V2,
    ));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn _download_dialogpt() -> anyhow::Result<()> {
    // Shared under MIT license by the Microsoft team at https://huggingface.co/microsoft/DialoGPT-medium. Modified with conversion to C-array format.
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        Gpt2ConfigResources::DIALOGPT_MEDIUM,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        Gpt2VocabResources::DIALOGPT_MEDIUM,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        Gpt2MergesResources::DIALOGPT_MEDIUM,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        Gpt2ModelResources::DIALOGPT_MEDIUM,
    ));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = merges_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_t5_small() -> anyhow::Result<()> {
    // Shared under Apache 2.0 license by the Google team at https://github.com/google-research/text-to-text-transfer-transformer.
    let config_resource =
        Resource::Remote(RemoteResource::from_pretrained(T5ConfigResources::T5_SMALL));
    let vocab_resource =
        Resource::Remote(RemoteResource::from_pretrained(T5VocabResources::T5_SMALL));
    let weights_resource =
        Resource::Remote(RemoteResource::from_pretrained(T5ModelResources::T5_SMALL));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_roberta_qa() -> anyhow::Result<()> {
    // Shared under Apache 2.0 license by [deepset](https://deepset.ai) at https://huggingface.co/deepset/roberta-base-squad2.
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaConfigResources::ROBERTA_QA,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaVocabResources::ROBERTA_QA,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaModelResources::ROBERTA_QA,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaMergesResources::ROBERTA_QA,
    ));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = merges_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_bert_qa() -> anyhow::Result<()> {
    // Shared under Apache 2.0 license by [deepset](https://deepset.ai) at https://huggingface.co/deepset/roberta-base-squad2.
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        BertConfigResources::BERT_QA,
    ));
    let vocab_resource =
        Resource::Remote(RemoteResource::from_pretrained(BertVocabResources::BERT_QA));
    let weights_resource =
        Resource::Remote(RemoteResource::from_pretrained(BertModelResources::BERT_QA));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_xlm_roberta_ner_german() -> anyhow::Result<()> {
    // Shared under Apache 2.0 license by the HuggingFace Inc. team at https://huggingface.co/models.
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaConfigResources::XLM_ROBERTA_NER_DE,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaVocabResources::XLM_ROBERTA_NER_DE,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        RobertaModelResources::XLM_ROBERTA_NER_DE,
    ));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn download_xlnet_base_cased() -> anyhow::Result<()> {
    // Shared under Apache 2.0 license by the HuggingFace Inc. team at https://huggingface.co/models.
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        XLNetConfigResources::XLNET_BASE_CASED,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        XLNetVocabResources::XLNET_BASE_CASED,
    ));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(
        XLNetModelResources::XLNET_BASE_CASED,
    ));
    let _ = config_resource.get_local_path()?;
    let _ = vocab_resource.get_local_path()?;
    let _ = weights_resource.get_local_path()?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let _ = download_distil_gpt2();
    let _ = download_distilbert_sst2();
    let _ = download_distilbert_qa();
    let _ = download_distilbert();
    let _ = download_gpt2();
    let _ = download_gpt();
    let _ = download_roberta();
    let _ = download_bert();
    let _ = download_bert_ner();
    let _ = download_bart();
    let _ = download_bart_cnn();
    let _ = download_electra_generator();
    let _ = download_electra_discriminator();
    let _ = download_albert_base_v2();
    let _ = download_t5_small();
    let _ = download_roberta_qa();
    let _ = download_bert_qa();
    let _ = download_xlm_roberta_ner_german();
    let _ = download_xlnet_base_cased();

    Ok(())
}
