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

extern crate failure;
extern crate dirs;

use std::path::PathBuf;
use tch::{Device, nn};
use rust_tokenizers::RobertaTokenizer;
use failure::err_msg;
use rust_bert::bart::BartConfig;
use rust_bert::Config;
use rust_bert::bart::bart::BartModel;


fn main() -> failure::Fallible<()> {
    //    Resources paths
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push("rustbert");
    home.push("bart-large-cnn");
    let config_path = &home.as_path().join("config.json");
    let vocab_path = &home.as_path().join("vocab.txt");
    let merges_path = &home.as_path().join("merges.txt");
    let weights_path = &home.as_path().join("model.ot");

    if !config_path.is_file() | !vocab_path.is_file() | !merges_path.is_file() | !weights_path.is_file() {
        return Err(
            err_msg("Could not find required resources to run example. \
                          Please run ../utils/download_dependencies_bart.py \
                          in a Python environment with dependencies listed in ../requirements.txt"));
    }

//    Set-up masked LM model
    let device = Device::Cpu;
    let mut vs = nn::VarStore::new(device);
    let _tokenizer = RobertaTokenizer::from_file(vocab_path.to_str().unwrap(), merges_path.to_str().unwrap(), false);
    let config = BartConfig::from_file(config_path);
    let bart_model = BartModel::new(&vs.root(), &config, false);
    vs.load(weights_path)?;
//
////    Define input
//    let input = ["New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
//    A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
//    Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared \"I do\" five more times, sometimes only within two weeks of each other. \
//    In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her \"first and only\" marriage. \
//    Barrientos, now 39, is facing two criminal counts of \"offering a false instrument for filing in the first degree,\" referring to her false statements on the
//    2010 marriage license application, according to court documents.
//    Prosecutors said the marriages were part of an immigration scam.
//    On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
//    After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
//    Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
//    All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
//    Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
//    Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
//    The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s \
//    Investigation Division. Seven of the men are from so-called \"red-flagged\" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
//    Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
//    If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18."];
//    let tokenized_input = tokenizer.encode_list(input.to_vec(), 1024, &TruncationStrategy::LongestFirst, 0);
//    let max_len = tokenized_input.iter().map(|input| input.token_ids.len()).max().unwrap();
//    let tokenized_input = tokenized_input.
//        iter().
//        map(|input| input.token_ids.clone()).
//        map(|mut input| {
//            input.extend(vec![0; max_len - input.len()]);
//            input
//        }).
//        map(|input|
//            Tensor::of_slice(&(input))).
//        collect::<Vec<_>>();
//    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);


////    Forward pass
//    let (output, _, _) = no_grad(|| {
//        bert_model
//            .forward_t(Some(input_tensor),
//                       None,
//                       None,
//                       None,
//                       None,
//                       &None,
//                       &None,
//                       false)
//    });
//
////    Print masked tokens
//    let index_1 = output.get(0).get(4).argmax(0, false);
//    let index_2 = output.get(1).get(7).argmax(0, false);
//    let word_1 = tokenizer.vocab().id_to_token(&index_1.int64_value(&[]));
//    let word_2 = tokenizer.vocab().id_to_token(&index_2.int64_value(&[]));
//
//    println!("{}", word_1); // Outputs "person" : "Looks like one [person] is missing"
//    println!("{}", word_2);// Outputs "pear" : "It was a very nice and [pleasant] day"

    Ok(())
}