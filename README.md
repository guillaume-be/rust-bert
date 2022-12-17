# rust-bert

[![Build Status](https://github.com/guillaume-be/rust-bert/workflows/Build/badge.svg?event=push)](https://github.com/guillaume-be/rust-bert/actions)
[![Latest version](https://img.shields.io/crates/v/rust_bert.svg)](https://crates.io/crates/rust_bert)
[![Documentation](https://docs.rs/rust-bert/badge.svg)](https://docs.rs/rust-bert)
![License](https://img.shields.io/crates/l/rust_bert.svg)

Rust-native state-of-the-art Natural Language Processing models and pipelines. Port of Hugging Face's [Transformers library](https://github.com/huggingface/transformers), using the [tch-rs](https://github.com/LaurentMazare/tch-rs) crate and pre-processing from [rust-tokenizers](https://github.com/guillaume-be/rust-tokenizers). Supports multi-threaded tokenization and GPU inference.
This repository exposes the model base architecture, task-specific heads (see below) and [ready-to-use pipelines](#ready-to-use-pipelines). [Benchmarks](#benchmarks) are available at the end of this document.

Get started with tasks including question answering, named entity recognition, translation, summarization, text generation, conversational agents and more in just a few lines of code:
```rust
    let qa_model = QuestionAnsweringModel::new(Default::default())?;
                                                        
    let question = String::from("Where does Amy live ?");
    let context = String::from("Amy lives in Amsterdam");

    let answers = qa_model.predict(&[QaInput { question, context }], 1, 32);
```

Output:
```
[Answer { score: 0.9976, start: 13, end: 21, answer: "Amsterdam" }]
```

The tasks currently supported include:
  - Translation
  - Summarization
  - Multi-turn dialogue
  - Zero-shot classification
  - Sentiment Analysis
  - Named Entity Recognition
  - Part of Speech tagging
  - Question-Answering
  - Language Generation
  - Masked Language Model
  - Sentence Embeddings

<details>
<summary> <b>Expand to display the supported models/tasks matrix </b> </summary>

| |**Sequence classification**|**Token classification**|**Question answering**|**Text Generation**|**Summarization**|**Translation**|**Masked LM**|**Sentence Embeddings**|
:-----:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:|:----:
DistilBERT|✅|✅|✅| | | |✅| ✅| 
MobileBERT|✅|✅|✅| | | |✅| |
DeBERTa|✅|✅|✅| | | |✅| |
DeBERTa (v2)|✅|✅|✅| | | |✅| |
FNet|✅|✅|✅| | | |✅| |
BERT|✅|✅|✅| | | |✅| ✅|
RoBERTa|✅|✅|✅| | | |✅| ✅| 
GPT| | | |✅ | | | |  |
GPT2| | | |✅ | | | |  |
GPT-Neo| | | |✅ | | | | | 
BART|✅| | |✅ |✅| | | |
Marian| | | |  | |✅| |  |
MBart|✅| | |✅ | | | |  |
M2M100| | | |✅ | | | |  |
Electra | |✅| | | | |✅|  |
ALBERT |✅|✅|✅| | | |✅| ✅ |
T5 | | | |✅ |✅|✅| | ✅ |
XLNet|✅|✅|✅|✅ | | |✅|  |
Reformer|✅| |✅|✅ | | |✅|  |
ProphetNet| | | |✅ |✅ | | |  |
Longformer|✅|✅|✅| | | |✅|  |
Pegasus| | | | |✅| | |  |
</details>

## Getting started

This library relies on the [tch](https://github.com/LaurentMazare/tch-rs) crate for bindings to the C++ Libtorch API.
The libtorch library is required can be downloaded either automatically or manually. The following provides a reference on how to set-up your environment
to use these bindings, please refer to the [tch](https://github.com/LaurentMazare/tch-rs) for detailed information or support.

Furthermore, this library relies on a cache folder for downloading pre-trained models. 
This cache location defaults to `~/.cache/.rustbert`, but can be changed by setting the `RUSTBERT_CACHE` environment variable. Note that the language models used by this library are in the order of the 100s of MBs to GBs.

### Manual installation (recommended)

1. Download `libtorch` from https://pytorch.org/get-started/locally/. This package requires `v1.13.0`: if this version is no longer available on the "get started" page,
the file should be accessible by modifying the target link, for example `https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcu117.zip` for a Linux version with CUDA11.
2. Extract the library to a location of your choice
3. Set the following environment variables
##### Linux:
```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```

##### Windows
```powershell
$Env:LIBTORCH = "X:\path\to\libtorch"
$Env:Path += ";X:\path\to\libtorch\lib"
```

### Automatic installation

Alternatively, you can let the `build` script automatically download the `libtorch` library for you.
The CPU version of libtorch will be downloaded by default. To download a CUDA version, please set the environment variable `TORCH_CUDA_VERSION` to `cu117`.
Note that the libtorch library is large (order of several GBs for the CUDA-enabled version) and the first build may therefore take several minutes to complete.

## Ready-to-use pipelines
	
Based on Hugging Face's pipelines, ready to use end-to-end NLP pipelines are available as part of this crate. The following capabilities are currently available:

**Disclaimer**
The contributors of this repository are not responsible for any generation from the 3rd party utilization of the pretrained systems proposed herein.

<details>
<summary> <b>1. Question Answering</b> </summary>

Extractive question answering from a given question and context. DistilBERT model fine-tuned on SQuAD (Stanford Question Answering Dataset)

```rust
    let qa_model = QuestionAnsweringModel::new(Default::default())?;
                                                        
    let question = String::from("Where does Amy live ?");
    let context = String::from("Amy lives in Amsterdam");

    let answers = qa_model.predict(&[QaInput { question, context }], 1, 32);
```

Output:
```
[Answer { score: 0.9976, start: 13, end: 21, answer: "Amsterdam" }]
```
</details>
&nbsp;  
<details>
<summary> <b>2. Translation </b> </summary>

Translation pipeline supporting a broad range of source and target languages. Leverages two main architectures for translation tasks:
- Marian-based models, for specific source/target combinations
- M2M100 models allowing for direct translation between 100 languages (at a higher computational cost and lower performance for some selected languages)

Marian-based pretrained models for the following language pairs are readily available in the library - but the user can import any Pytorch-based
model for predictions
- English <-> French
- English <-> Spanish
- English <-> Portuguese
- English <-> Italian
- English <-> Catalan
- English <-> German
- English <-> Russian
- English <-> Chinese
- English <-> Dutch
- English <-> Swedish
- English <-> Arabic
- English <-> Hebrew
- English <-> Hindi
- French <-> German

For languages not supported by the proposed pretrained Marian models, the user can leverage a M2M100 model supporting direct translation between 100 languages (without intermediate English translation)
The full list of supported languages is available in the [crate documentation](https://docs.rs/rust-bert/latest/rust_bert/pipelines/translation/enum.Language.html)

 ```rust
 use rust_bert::pipelines::translation::{Language, TranslationModelBuilder};
 fn main() -> anyhow::Result<()> {
 let model = TranslationModelBuilder::new()
         .with_source_languages(vec![Language::English])
         .with_target_languages(vec![Language::Spanish, Language::French, Language::Italian])
         .create_model()?;
     let input_text = "This is a sentence to be translated";
     let output = model.translate(&[input_text], None, Language::Spanish)?;
     for sentence in output {
         println!("{}", sentence);
     }
     Ok(())
 }
 ```
Output:
```
Il s'agit d'une phrase à traduire
```
</details>
&nbsp;  
<details>
<summary> <b>3. Summarization </b> </summary>

Abstractive summarization using a pretrained BART model.

```rust
    let summarization_model = SummarizationModel::new(Default::default())?;
                                                        
    let input = ["In findings published Tuesday in Cornell University's arXiv by a team of scientists \
from the University of Montreal and a separate report published Wednesday in Nature Astronomy by a team \
from University College London (UCL), the presence of water vapour was confirmed in the atmosphere of K2-18b, \
a planet circling a star in the constellation Leo. This is the first such discovery in a planet in its star's \
habitable zone — not too hot and not too cold for liquid water to exist. The Montreal team, led by Björn Benneke, \
used data from the NASA's Hubble telescope to assess changes in the light coming from K2-18b's star as the planet \
passed between it and Earth. They found that certain wavelengths of light, which are usually absorbed by water, \
weakened when the planet was in the way, indicating not only does K2-18b have an atmosphere, but the atmosphere \
contains water in vapour form. The team from UCL then analyzed the Montreal team's data using their own software \
and confirmed their conclusion. This was not the first time scientists have found signs of water on an exoplanet, \
but previous discoveries were made on planets with high temperatures or other pronounced differences from Earth. \
\"This is the first potentially habitable planet where the temperature is right and where we now know there is water,\" \
said UCL astronomer Angelos Tsiaras. \"It's the best candidate for habitability right now.\" \"It's a good sign\", \
said Ryan Cloutier of the Harvard–Smithsonian Center for Astrophysics, who was not one of either study's authors. \
\"Overall,\" he continued, \"the presence of water in its atmosphere certainly improves the prospect of K2-18b being \
a potentially habitable planet, but further observations will be required to say for sure. \"
K2-18b was first identified in 2015 by the Kepler space telescope. It is about 110 light-years from Earth and larger \
but less dense. Its star, a red dwarf, is cooler than the Sun, but the planet's orbit is much closer, such that a year \
on K2-18b lasts 33 Earth days. According to The Guardian, astronomers were optimistic that NASA's James Webb space \
telescope — scheduled for launch in 2021 — and the European Space Agency's 2028 ARIEL program, could reveal more \
about exoplanets like K2-18b."];

    let output = summarization_model.summarize(&input);
```
(example from: [WikiNews](https://en.wikinews.org/wiki/Astronomers_find_water_vapour_in_atmosphere_of_exoplanet_K2-18b))

Output:
```
"Scientists have found water vapour on K2-18b, a planet 110 light-years from Earth. 
This is the first such discovery in a planet in its star's habitable zone. 
The planet is not too hot and not too cold for liquid water to exist."
```
</details>
&nbsp;  
<details>
<summary> <b>4. Dialogue Model </b> </summary>

Conversation model based on Microsoft's [DialoGPT](https://github.com/microsoft/DialoGPT).
This pipeline allows the generation of single or multi-turn conversations between a human and a model.
The DialoGPT's page states that
> The human evaluation results indicate that the response generated from DialoGPT is comparable to human response quality
> under a single-turn conversation Turing test. ([DialoGPT repository](https://github.com/microsoft/DialoGPT))

The model uses a `ConversationManager` to keep track of active conversations and generate responses to them.

```rust
use rust_bert::pipelines::conversation::{ConversationModel, ConversationManager};

let conversation_model = ConversationModel::new(Default::default());
let mut conversation_manager = ConversationManager::new();

let conversation_id = conversation_manager.create("Going to the movies tonight - any suggestions?");
let output = conversation_model.generate_responses(&mut conversation_manager);
```
Example output:
```
"The Big Lebowski."
```
</details>
&nbsp;  
<details>
<summary> <b>5. Natural Language Generation </b> </summary>

Generate language based on a prompt. GPT2 and GPT available as base models.
Include techniques such as beam search, top-k and nucleus sampling, temperature setting and repetition penalty.
Supports batch generation of sentences from several prompts. Sequences will be left-padded with the model's padding token if present, the unknown token otherwise.
This may impact the results, it is recommended to submit prompts of similar length for best results

```rust
    let model = GPT2Generator::new(Default::default())?;
                                                        
    let input_context_1 = "The dog";
    let input_context_2 = "The cat was";

    let generate_options = GenerateOptions {
        max_length: 30,
        ..Default::default()
    };

    let output = model.generate(Some(&[input_context_1, input_context_2]), generate_options);
```
Example output:
```
[
    "The dog's owners, however, did not want to be named. According to the lawsuit, the animal's owner, a 29-year"
    "The dog has always been part of the family. \"He was always going to be my dog and he was always looking out for me"
    "The dog has been able to stay in the home for more than three months now. \"It's a very good dog. She's"
    "The cat was discovered earlier this month in the home of a relative of the deceased. The cat\'s owner, who wished to remain anonymous,"
    "The cat was pulled from the street by two-year-old Jazmine.\"I didn't know what to do,\" she said"
    "The cat was attacked by two stray dogs and was taken to a hospital. Two other cats were also injured in the attack and are being treated."
]
```
</details>
&nbsp;  
<details>
<summary> <b>6. Zero-shot classification </b> </summary>

Performs zero-shot classification on input sentences with provided labels using a model fine-tuned for Natural Language Inference.
```rust
    let sequence_classification_model = ZeroShotClassificationModel::new(Default::default())?;

    let input_sentence = "Who are you voting for in 2020?";
    let input_sequence_2 = "The prime minister has announced a stimulus package which was widely criticized by the opposition.";
    let candidate_labels = &["politics", "public health", "economics", "sports"];

    let output = sequence_classification_model.predict_multilabel(
        &[input_sentence, input_sequence_2],
        candidate_labels,
        None,
        128,
    );
```

Output:
```
[
  [ Label { "politics", score: 0.972 }, Label { "public health", score: 0.032 }, Label {"economics", score: 0.006 }, Label {"sports", score: 0.004 } ],
  [ Label { "politics", score: 0.975 }, Label { "public health", score: 0.0818 }, Label {"economics", score: 0.852 }, Label {"sports", score: 0.001 } ],
]
```
</details>
&nbsp;  
<details>
<summary> <b>7. Sentiment analysis </b> </summary>

Predicts the binary sentiment for a sentence. DistilBERT model fine-tuned on SST-2.
```rust
    let sentiment_classifier = SentimentModel::new(Default::default())?;
                                                        
    let input = [
        "Probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring.",
        "This film tried to be too many things all at once: stinging political satire, Hollywood blockbuster, sappy romantic comedy, family values promo...",
        "If you like original gut wrenching laughter you will like this movie. If you are young or old then you will love this movie, hell even my mom liked it.",
    ];

    let output = sentiment_classifier.predict(&input);
```
(Example courtesy of [IMDb](http://www.imdb.com))

Output:
```
[
    Sentiment { polarity: Positive, score: 0.9981985493795946 },
    Sentiment { polarity: Negative, score: 0.9927982091903687 },
    Sentiment { polarity: Positive, score: 0.9997248985164333 }
]
```
</details>
&nbsp;  
<details>
<summary> <b>8. Named Entity Recognition </b> </summary>

Extracts entities (Person, Location, Organization, Miscellaneous) from text. BERT cased large model fine-tuned on CoNNL03, contributed by the [MDZ Digital Library team at the Bavarian State Library](https://github.com/dbmdz).
Models are currently available for English, German, Spanish and Dutch.
```rust
    let ner_model = NERModel::new(default::default())?;

    let input = [
        "My name is Amy. I live in Paris.",
        "Paris is a city in France."
    ];
    
    let output = ner_model.predict(&input);
```
Output:
```
[
  [
    Entity { word: "Amy", score: 0.9986, label: "I-PER" }
    Entity { word: "Paris", score: 0.9985, label: "I-LOC" }
  ],
  [
    Entity { word: "Paris", score: 0.9988, label: "I-LOC" }
    Entity { word: "France", score: 0.9993, label: "I-LOC" }
  ]
]
```
</details>
&nbsp;  
<details>
<summary> <b>9. Keywords/keyphrases extraction</b> </summary>

Extract keywords and keyphrases extractions from input documents

```rust
fn main() -> anyhow::Result<()> {
    let keyword_extraction_model = KeywordExtractionModel::new(Default::default())?;
    
    let input = "Rust is a multi-paradigm, general-purpose programming language. \
       Rust emphasizes performance, type safety, and concurrency. Rust enforces memory safety—that is, \
       that all references point to valid memory—without requiring the use of a garbage collector or \
       reference counting present in other memory-safe languages. To simultaneously enforce \
       memory safety and prevent concurrent data races, Rust's borrow checker tracks the object lifetime \
       and variable scope of all references in a program during compilation. Rust is popular for \
       systems programming but also offers high-level features including functional programming constructs.";

    let output = keyword_extraction_model.predict(&[input])?;
}
```
Output:
```
"rust" - 0.50910604
"programming" - 0.35731024
"concurrency" - 0.33825397
"concurrent" - 0.31229728
"program" - 0.29115444
```
</details>
&nbsp;  
<details>
<summary> <b>10. Part of Speech tagging </b> </summary>

Extracts Part of Speech tags (Noun, Verb, Adjective...) from text.
```rust
    let pos_model = POSModel::new(default::default())?;

    let input = ["My name is Bob"];
    
    let output = pos_model.predict(&input);
```
Output:
```
[
    Entity { word: "My", score: 0.1560, label: "PRP" }
    Entity { word: "name", score: 0.6565, label: "NN" }
    Entity { word: "is", score: 0.3697, label: "VBZ" }
    Entity { word: "Bob", score: 0.7460, label: "NNP" }
]
```
</details>
&nbsp;  
<details>
<summary> <b>11. Sentence embeddings </b> </summary>

Generate sentence embeddings (vector representation). These can be used for applications including dense information retrieval.
```rust
    let model = SentenceEmbeddingsBuilder::remote(
            SentenceEmbeddingsModelType::AllMiniLmL12V2
        ).create_model()?;

    let sentences = [
        "this is an example sentence", 
        "each sentence is converted"
    ];
    
    let output = model.predict(&sentences);
```
Output:
```
[
    [-0.000202666, 0.08148022, 0.03136178, 0.002920636 ...],
    [0.064757116, 0.048519745, -0.01786038, -0.0479775 ...]
]
```
</details>


<details>
<summary> <b>12. Masked Language Model </b> </summary>

Predict masked words in input sentences.
```rust
    let model = MaskedLanguageModel::new(Default::default())?;

    let sentences = [
        "Hello I am a <mask> student",
        "Paris is the <mask> of France. It is <mask> in Europe.",
    ];
    
    let output = model.predict(&sentences);
```
Output:
```
[
    [MaskedToken { text: "college", id: 2267, score: 8.091}],
    [
        MaskedToken { text: "capital", id: 3007, score: 16.7249}, 
        MaskedToken { text: "located", id: 2284, score: 9.0452}
    ]
]
```
</details>

## Benchmarks

For simple pipelines (sequence classification, tokens classification, question answering) the performance between Python and Rust is expected to be comparable. This is because the most expensive part of these pipeline is the language model itself, sharing a common implementation in the Torch backend. The [End-to-end NLP Pipelines in Rust](https://www.aclweb.org/anthology/2020.nlposs-1.4/) provides a benchmarks section covering all pipelines.

For text generation tasks (summarization, translation, conversation, free text generation), significant benefits can be expected (up to 2 to 4 times faster processing depending on the input and application). The article [Accelerating text generation with Rust](https://guillaume-be.github.io/2020-11-21/generation_benchmarks) focuses on these text generation applications and provides more details on the performance comparison to Python.

## Loading pretrained and custom model weights

The base model and task-specific heads are also available for users looking to expose their own transformer based models.
Examples on how to prepare the date using a native tokenizers Rust library are available in `./examples` for BERT, DistilBERT, RoBERTa, GPT, GPT2 and BART.
Note that when importing models from Pytorch, the convention for parameters naming needs to be aligned with the Rust schema. Loading of the pre-trained weights will fail if any of the model parameters weights cannot be found in the weight files.
If this quality check is to be skipped, an alternative method `load_partial` can be invoked from the variables store.

Pretrained models are available on Hugging face's [model hub](https://huggingface.co/models?filter=rust) and can be loaded using `RemoteResources` defined in this library.
A conversion utility script is included in `./utils` to convert Pytorch weights to a set of weights compatible with this library. This script requires Python and `torch` to be set-up, and can be used as follows:
`python ./utils/convert_model.py path/to/pytorch_model.bin` where `path/to/pytorch_model.bin` is the location of the original Pytorch weights.


## Citation

If you use `rust-bert` for your work, please cite [End-to-end NLP Pipelines in Rust](https://www.aclweb.org/anthology/2020.nlposs-1.4/):
```bibtex
@inproceedings{becquin-2020-end,
    title = "End-to-end {NLP} Pipelines in Rust",
    author = "Becquin, Guillaume",
    booktitle = "Proceedings of Second Workshop for NLP Open Source Software (NLP-OSS)",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.nlposs-1.4",
    pages = "20--25",
}
```

## Acknowledgements

Thank you to [Hugging Face](https://huggingface.co) for hosting a set of weights compatible with this Rust library.
The list of ready-to-use pretrained models is listed at [https://huggingface.co/models?filter=rust](https://huggingface.co/models?filter=rust).

