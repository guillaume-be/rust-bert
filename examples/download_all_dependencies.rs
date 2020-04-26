extern crate failure;

use rust_bert::gpt2::{Gpt2ConfigResources, Gpt2VocabResources, Gpt2MergesResources, Gpt2ModelResources};
use rust_bert::distilbert::{DistilBertModelResources, DistilBertConfigResources, DistilBertVocabResources};
use rust_bert::openai_gpt::{OpenAiGptConfigResources, OpenAiGptVocabResources, OpenAiGptMergesResources, OpenAiGptModelResources};
use rust_bert::roberta::{RobertaConfigResources, RobertaVocabResources, RobertaMergesResources, RobertaModelResources};
use rust_bert::bert::{BertConfigResources, BertVocabResources, BertModelResources};
use rust_bert::bart::{BartConfigResources, BartVocabResources, BartMergesResources, BartModelResources};
use rust_bert::resources::{Resource, download_resource, RemoteResource};

/// This example downloads and caches all dependencies used in model tests. This allows for safe
/// multi threaded testing (two test using the same resource would otherwise download the file to
/// the same location).


fn download_distil_gpt2() -> failure::Fallible<()> {
//   Shared under Apache 2.0 license by the HuggingFace Inc. team at https://huggingface.co/models
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(Gpt2ConfigResources::DISTIL_GPT2));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(Gpt2VocabResources::DISTIL_GPT2));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(Gpt2MergesResources::DISTIL_GPT2));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(Gpt2ModelResources::DISTIL_GPT2));
    let _ = download_resource(&config_resource)?;
    let _ = download_resource(&vocab_resource)?;
    let _ = download_resource(&merges_resource)?;
    let _ = download_resource(&weights_resource)?;
    Ok(())
}

fn download_distilbert_sst2() -> failure::Fallible<()> {
//   Shared under Apache 2.0 license by the HuggingFace Inc. team at https://huggingface.co/models
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(DistilBertModelResources::DISTIL_BERT_SST2));
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(DistilBertConfigResources::DISTIL_BERT_SST2));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(DistilBertVocabResources::DISTIL_BERT_SST2));
    let _ = download_resource(&config_resource)?;
    let _ = download_resource(&vocab_resource)?;
    let _ = download_resource(&weights_resource)?;
    Ok(())
}

fn download_distilbert_qa() -> failure::Fallible<()> {
//   Shared under Apache 2.0 license by the HuggingFace Inc. team at https://huggingface.co/models
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(DistilBertModelResources::DISTIL_BERT_SQUAD));
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(DistilBertConfigResources::DISTIL_BERT_SQUAD));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(DistilBertVocabResources::DISTIL_BERT_SQUAD));
    let _ = download_resource(&config_resource)?;
    let _ = download_resource(&vocab_resource)?;
    let _ = download_resource(&weights_resource)?;
    Ok(())
}

fn download_distilbert() -> failure::Fallible<()> {
//   Shared under Apache 2.0 license by the HuggingFace Inc. team at https://huggingface.co/models
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(DistilBertModelResources::DISTIL_BERT));
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(DistilBertConfigResources::DISTIL_BERT));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(DistilBertVocabResources::DISTIL_BERT));
    let _ = download_resource(&config_resource)?;
    let _ = download_resource(&vocab_resource)?;
    let _ = download_resource(&weights_resource)?;
    Ok(())
}

fn download_gpt2() -> failure::Fallible<()> {
//   Shared under Modified MIT license by the OpenAI team at https://github.com/openai/gpt-2
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));
    let _ = download_resource(&config_resource)?;
    let _ = download_resource(&vocab_resource)?;
    let _ = download_resource(&merges_resource)?;
    let _ = download_resource(&weights_resource)?;
    Ok(())
}

fn download_gpt() -> failure::Fallible<()> {
//   Shared under MIT license by the OpenAI team at https://github.com/openai/finetune-transformer-lm
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptConfigResources::GPT));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptVocabResources::GPT));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptMergesResources::GPT));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(OpenAiGptModelResources::GPT));
    let _ = download_resource(&config_resource)?;
    let _ = download_resource(&vocab_resource)?;
    let _ = download_resource(&merges_resource)?;
    let _ = download_resource(&weights_resource)?;
    Ok(())
}

fn download_roberta() -> failure::Fallible<()> {
//   Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(RobertaConfigResources::ROBERTA));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(RobertaVocabResources::ROBERTA));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(RobertaMergesResources::ROBERTA));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(RobertaModelResources::ROBERTA));
    let _ = download_resource(&config_resource)?;
    let _ = download_resource(&vocab_resource)?;
    let _ = download_resource(&merges_resource)?;
    let _ = download_resource(&weights_resource)?;
    Ok(())
}

fn download_bert() -> failure::Fallible<()> {
//   Shared under Apache 2.0 license by the Google team at https://github.com/google-research/bert
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(BertConfigResources::BERT));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(BertVocabResources::BERT));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(BertModelResources::BERT));
    let _ = download_resource(&config_resource)?;
    let _ = download_resource(&vocab_resource)?;
    let _ = download_resource(&weights_resource)?;
    Ok(())
}

fn download_bert_ner() -> failure::Fallible<()> {
//    Shared under MIT license by the MDZ Digital Library team at the Bavarian State Library at https://github.com/dbmdz/berts
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(BertConfigResources::BERT_NER));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(BertVocabResources::BERT_NER));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(BertModelResources::BERT_NER));
    let _ = download_resource(&config_resource)?;
    let _ = download_resource(&vocab_resource)?;
    let _ = download_resource(&weights_resource)?;
    Ok(())
}

fn download_bart() -> failure::Fallible<()> {
//   Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(BartConfigResources::BART));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(BartVocabResources::BART));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(BartMergesResources::BART));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(BartModelResources::BART));
    let _ = download_resource(&config_resource)?;
    let _ = download_resource(&vocab_resource)?;
    let _ = download_resource(&merges_resource)?;
    let _ = download_resource(&weights_resource)?;
    Ok(())
}

fn download_bart_cnn() -> failure::Fallible<()> {
//   Shared under MIT license by the Facebook AI Research Fairseq team at https://github.com/pytorch/fairseq
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(BartConfigResources::BART_CNN));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(BartVocabResources::BART_CNN));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(BartMergesResources::BART_CNN));
    let weights_resource = Resource::Remote(RemoteResource::from_pretrained(BartModelResources::BART_CNN));
    let _ = download_resource(&config_resource)?;
    let _ = download_resource(&vocab_resource)?;
    let _ = download_resource(&merges_resource)?;
    let _ = download_resource(&weights_resource)?;
    Ok(())
}

fn main() -> failure::Fallible<()> {
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

    Ok(())
}