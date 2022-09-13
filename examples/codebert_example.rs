// Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
// Copyright 2019 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use anyhow::Result;
use rust_bert::codebert::{
    CodeBertConfigResources, CodeBertForFeatureExtractionConfig, CodeBertForFeatureExtractionModel,
    CodeBertMergesResources, CodeBertModelResources, CodeBertVocabResources,
};
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::masked_language::{MaskedLanguageConfig, MaskedLanguageModel};
use rust_bert::pipelines::sequence_classification::{
    SequenceClassificationConfig, SequenceClassificationModel,
};
use rust_bert::resources::{LocalResource, RemoteResource};
use std::path::PathBuf;
fn feature_extraction() -> Result<()> {
    //    Set-up model
    let config = CodeBertForFeatureExtractionConfig::new(
        ModelType::CodeBert,
        RemoteResource::from_pretrained(CodeBertModelResources::CODEBERT),
        None,
        RemoteResource::from_pretrained(CodeBertConfigResources::CODEBERT),
        RemoteResource::from_pretrained(CodeBertVocabResources::CODEBERT),
        RemoteResource::from_pretrained(CodeBertMergesResources::CODEBERT),
        true,
        None,
        None,
    );

    let feature_extraction_model = CodeBertForFeatureExtractionModel::new(config)?;
    //    Define input
    let input = ["this is an example sentence", "each sentence is converted"];

    //    Run model
    let output = feature_extraction_model.predict(&input);
    for hidden_states in output {
        println!("{:?}", hidden_states);
    }

    Ok(())
}

fn masked_language() -> Result<()> {
    let config = MaskedLanguageConfig::new(
        ModelType::CodeBert,
        RemoteResource::from_pretrained(CodeBertModelResources::CODEBERT_MLM),
        // None,
        LocalResource {
            local_path: PathBuf::from("/home/vincent/.cache/model/codebert-base-mlm/rust_model.ot"),
        },
        RemoteResource::from_pretrained(CodeBertConfigResources::CODEBERT_MLM),
        RemoteResource::from_pretrained(CodeBertVocabResources::CODEBERT_MLM),
        Some(RemoteResource::from_pretrained(
            CodeBertMergesResources::CODEBERT_MLM,
        )),
        true,
        None,
        false,
    );
    //    Set-up model
    let model = MaskedLanguageModel::new(config)?;
    //    Define input
    let input = [
        "Looks like one <mask> is missing!",
        "The goal of life is <mask>.",
    ];

    //    Run model
    let output = model.predict(&input, vec![5, 6]);
    for word in output {
        println!("{:?}", word);
    }

    Ok(())
}

fn sequence_classification() -> Result<()> {
    let config = SequenceClassificationConfig::new(
        ModelType::CodeBert,
        RemoteResource::from_pretrained(CodeBertModelResources::CODEBERTA_LANG),
        None,
        RemoteResource::from_pretrained(CodeBertConfigResources::CODEBERTA_LANG),
        RemoteResource::from_pretrained(CodeBertVocabResources::CODEBERTA_LANG),
        RemoteResource::from_pretrained(CodeBertMergesResources::CODEBERTA_LANG),
        true,
        None,
        false,
    );
    //    Set-up model
    let sequence_classification_model = SequenceClassificationModel::new(config)?;
    //    Define input
    let input = [
        "const foo = 'bar'",
        "Toutcome := rand.Intn(6) + 1",
        "def f(x): return x**2",
    ];

    //    Run model
    let output = sequence_classification_model.predict(&input);
    for label in output {
        println!("{:?}", label);
    }

    Ok(())
}

fn main() {
    feature_extraction().ok();
    masked_language().ok();
    sequence_classification().ok();
}
