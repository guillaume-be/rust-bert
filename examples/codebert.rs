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

extern crate anyhow;

use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::masked_language::{MaskedLanguageConfig, MaskedLanguageModel};
use rust_bert::pipelines::sequence_classification::{
    SequenceClassificationConfig, SequenceClassificationModel,
};
use rust_bert::resources::RemoteResource;
use rust_bert::roberta::{
    RobertaConfigResources, RobertaMergesResources, RobertaModelResources, RobertaVocabResources,
};

fn main() -> anyhow::Result<()> {
    //    Language identification
    let sequence_classification_config = SequenceClassificationConfig::new(
        ModelType::Roberta,
        RemoteResource::from_pretrained(RobertaModelResources::CODEBERTA_LANGUAGE_ID),
        RemoteResource::from_pretrained(RobertaConfigResources::CODEBERTA_LANGUAGE_ID),
        RemoteResource::from_pretrained(RobertaVocabResources::CODEBERTA_LANGUAGE_ID),
        Some(RemoteResource::from_pretrained(
            RobertaMergesResources::CODEBERTA_LANGUAGE_ID,
        )),
        false,
        None,
        None,
    );

    let sequence_classification_model =
        SequenceClassificationModel::new(sequence_classification_config)?;

    //    Define input
    let input = [
        "def f(x):\
            return x**2",
        "outcome := rand.Intn(6) + 1",
    ];

    //    Run model
    let output = sequence_classification_model.predict(input);
    for label in output {
        println!("{label:?}");
    }

    // Masked language model
    let config = MaskedLanguageConfig::new(
        ModelType::Roberta,
        RemoteResource::from_pretrained(RobertaModelResources::CODEBERT_MLM),
        RemoteResource::from_pretrained(RobertaConfigResources::CODEBERT_MLM),
        RemoteResource::from_pretrained(RobertaVocabResources::CODEBERT_MLM),
        Some(RemoteResource::from_pretrained(
            RobertaMergesResources::CODEBERT_MLM,
        )),
        false,
        None,
        None,
        Some(String::from("<mask>")),
    );

    let mask_language_model = MaskedLanguageModel::new(config)?;
    //    Define input
    let input = [
        "if (x is not None) <mask> (x>1)",
        "<mask> x = if let <mask>(x_option) {}",
    ];

    //    Run model
    let output = mask_language_model.predict(input)?;
    for sentence_output in output {
        println!("{sentence_output:?}");
    }

    Ok(())
}
