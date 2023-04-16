// Copyright 2019-2020 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use tch::Device;

use crate::common::error::RustBertError;
use crate::m2m_100::M2M100Generator;
use crate::marian::MarianGenerator;
use crate::mbart::MBartGenerator;
use crate::nllb::NLLBGenerator;
use crate::pipelines::common::{ModelType, TokenizerOption};
use crate::pipelines::generation_utils::private_generation_utils::PrivateLanguageGenerator;
use crate::pipelines::generation_utils::{GenerateConfig, GenerateOptions, LanguageGenerator};
use crate::resources::ResourceProvider;
use crate::t5::T5Generator;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::fmt::{Debug, Display};

/// Language
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Language {
    Latvian,               // lv lav
    Achinese,              // ace
    MesopotamianArabic,    // acm
    TaizziAdeniArabic,     // acq
    TunisianArabic,        // aeb
    Afrikaans,             // afr af
    SouthLevantineArabic,  // ajp
    Akan,                  // aka ak
    Amharic,               // amh am
    NorthLevantineArabic,  // apc
    NajdiArabic,           // ars
    MoroccanArabic,        // ary
    EgyptianArabic,        // arz
    Assamese,              // asm as
    Asturian,              // ast
    Awadhi,                // awa
    CentralAymara,         // ayr
    SouthAzerbaijani,      // azb
    NorthAzerbaijani,      // azj
    Bashkir,               // bak ba
    Bambara,               // bam bm
    Balinese,              // ban
    Belarusian,            // bel be
    Bemba,                 // bem
    Bengali,               // ben bn
    Bhojpuri,              // bho
    Banjar,                // bjn
    Tibetan,               // bod bo
    Bosnian,               // bos bs
    Buginese,              // bug
    Bulgarian,             // bul bg
    Catalan,               // cat ca
    Cebuano,               // ceb
    Czech,                 // ces cs
    Chokwe,                // cjk
    CentralKurdish,        // ckb
    CrimeanTatar,          // crh
    Welsh,                 // cym cy
    Danish,                // dan da
    German,                // deu de
    SouthwesternDinka,     // dik
    Dyula,                 // dyu
    Dzongkha,              // dzo dz
    Greek,                 // ell el
    English,               // eng en
    Esperanto,             // epo eo
    Estonian,              // est et
    Basque,                // eus eu
    Ewe,                   // ewe ee
    Faroese,               // fao fo
    Fijian,                // fij fj
    Finnish,               // fin fi
    Fon,                   // fon
    French,                // fra fr
    Friulian,              // fur
    NigerianFulfulde,      // fuv
    WestCentralOromo,      // gaz
    ScottishGaelic,        // gla gd
    Irish,                 // gle ga
    Galician,              // glg gl
    Guarani,               // grn gn
    Gujarati,              // guj gu
    Haitian,               // hat ht
    Hausa,                 // hau ha
    Hebrew,                // heb he
    Hindi,                 // hin hi
    Chhattisgarhi,         // hne
    Croatian,              // hrv hr
    Hungarian,             // hun hu
    Armenian,              // hye hy
    Igbo,                  // ibo ig
    Iloko,                 // ilo
    Indonesian,            // ind id
    Icelandic,             // isl is
    Italian,               // ita it
    Javanese,              // jav jv
    Japanese,              // jpn ja
    Kabyle,                // kab
    Kachin,                // kac
    Kamba,                 // kam
    Kannada,               // kan kn
    Kashmiri,              // kas ks
    Georgian,              // kat ka
    Kazakh,                // kaz kk
    Kabiye,                // kbp
    Kabuverdianu,          // kea
    HalhMongolian,         // khk
    Khmer,                 // khm km
    Kikuyu,                // kik ki
    Kinyarwanda,           // kin rw
    Kirghiz,               // kir ky
    Kimbundu,              // kmb
    NorthernKurdish,       // kmr
    CentralKanuri,         // knc
    Kongo,                 // kon kg
    Korean,                // kor ko
    Lao,                   // lao lo
    Ligurian,              // lij
    Limburgan,             // lim li
    Lingala,               // lin ln
    Lithuanian,            // lit lt
    Lombard,               // lmo
    Latgalian,             // ltg
    Luxembourgish,         // ltz lb
    LubaLulua,             // lua
    Ganda,                 // lug lg
    Luo,                   // luo
    Lushai,                // lus
    Magahi,                // mag
    Maithili,              // mai
    Malayalam,             // mal ml
    Marathi,               // mar mr
    Minangkabau,           // min
    Macedonian,            // mkd mk
    Maltese,               // mlt mt
    Manipuri,              // mni
    Mossi,                 // mos
    Maori,                 // mri mi
    Burmese,               // mya my
    Dutch,                 // nld nl
    Norwegian,             // no
    NorwegianNynorsk,      // nno nn
    NorwegianBokmal,       // nob nb
    Nepali,                // npi
    Pedi,                  // nso
    Nuer,                  // nus
    Nyanja,                // nya ny
    Occitan,               // oci oc
    Odia,                  // ory
    Pangasinan,            // pag
    Panjabi,               // pan pa
    Papiamento,            // pap
    SouthernPashto,        // pbt
    IranianPersian,        // pes
    PlateauMalagasy,       // plt
    Polish,                // pol pl
    Portuguese,            // por pt
    Dari,                  // prs
    AyacuchoQuechua,       // quy
    Romanian,              // ron ro
    Rundi,                 // run rn
    Russian,               // rus ru
    Sango,                 // sag sg
    Sanskrit,              // san sa
    Santali,               // sat
    Sicilian,              // scn
    Shan,                  // shn
    Sinhala,               // sin si
    Slovak,                // slk sk
    Slovenian,             // slv sl
    Samoan,                // smo sm
    Shona,                 // sna sn
    Sindhi,                // snd sd
    Somali,                // som so
    SouthernSotho,         // sot st
    Spanish,               // spa es
    Sardinian,             // srd sc
    Serbian,               // srp sr
    Swati,                 // ssw ss
    Sundanese,             // sun su
    Swedish,               // swe sv
    Swahili,               // swh
    Silesian,              // szl
    Tamil,                 // tam ta
    Tamasheq,              // taq
    Tatar,                 // tat tt
    Telugu,                // tel te
    Tajik,                 // tgk tg
    Tagalog,               // tgl tl
    Thai,                  // tha th
    Tigrinya,              // tir ti
    TokPisin,              // tpi
    Tswana,                // tsn tn
    Tsonga,                // tso ts
    Turkmen,               // tuk tk
    Tumbuka,               // tum
    Turkish,               // tur tr
    Twi,                   // twi tw
    CentralAtlasTamazight, // tzm
    Uighur,                // uig ug
    Ukrainian,             // ukr uk
    Umbundu,               // umb
    Urdu,                  // urd ur
    NorthernUzbek,         // uzn
    Venetian,              // vec
    Vietnamese,            // vie vi
    Waray,                 // war
    Wolof,                 // wol wo
    Xhosa,                 // xho xh
    EasternYiddish,        // ydd
    Yoruba,                // yor yo
    YueChinese,            // yue
    Chinese,               // zho zh
    Zulu,                  // zul zu
    WesternFrisian,        // fy
    Arabic,                // ara ar
    Mongolian,             // mn
    Yiddish,               // yid yi
    Pashto,                // ps
    Farsi,
    Fulah,
    Uzbek,
    Malagasy,
    Albanian,
    Breton,
    Malay,
    Oriya,
    NorthernSotho,
    Luganda,
    Azerbaijani,
    ChineseMandarin,
    HaitianCreole,
    CentralKhmer,
}

impl Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", {
            let input_string = format!("{self:?}");
            let mut output: Vec<&str> = Vec::new();
            let mut start: usize = 0;

            for (c_pos, c) in input_string.char_indices() {
                if c.is_uppercase() {
                    if start < c_pos {
                        output.push(&input_string[start..c_pos]);
                    }
                    start = c_pos;
                }
            }
            if start < input_string.len() {
                output.push(&input_string[start..]);
            }
            output.join(" ")
        })
    }
}

impl Language {
    pub fn get_iso_639_1_code(&self) -> Option<&'static str> {
        let code = match self {
            Language::Afrikaans => "af",
            Language::Danish => "da",
            Language::Dutch => "nl",
            Language::German => "de",
            Language::English => "en",
            Language::Icelandic => "is",
            Language::Luxembourgish => "lb",
            Language::Norwegian => "no",
            Language::Swedish => "sv",
            Language::WesternFrisian => "fy",
            Language::Yiddish => "yi",
            Language::Asturian => "ast",
            Language::Catalan => "ca",
            Language::French => "fr",
            Language::Galician => "gl",
            Language::Italian => "it",
            Language::Occitan => "oc",
            Language::Portuguese => "pt",
            Language::Romanian => "ro",
            Language::Spanish => "es",
            Language::Belarusian => "be",
            Language::Bosnian => "bs",
            Language::Bulgarian => "bg",
            Language::Croatian => "hr",
            Language::Czech => "cs",
            Language::Macedonian => "mk",
            Language::Polish => "pl",
            Language::Russian => "ru",
            Language::Serbian => "sr",
            Language::Slovak => "sk",
            Language::Slovenian => "sl",
            Language::Ukrainian => "uk",
            Language::Estonian => "et",
            Language::Finnish => "fi",
            Language::Hungarian => "hu",
            Language::Lithuanian => "lt",
            Language::Armenian => "hy",
            Language::Georgian => "ka",
            Language::Greek => "el",
            Language::Breton => "br",
            Language::Irish => "ga",
            Language::ScottishGaelic => "gd",
            Language::Welsh => "cy",
            Language::NorthAzerbaijani => "az",
            Language::Bashkir => "ba",
            Language::Kazakh => "kk",
            Language::Turkish => "tr",
            Language::Uzbek => "uz",
            Language::NorthernUzbek => "uzn",
            Language::Japanese => "ja",
            Language::Korean => "ko",
            Language::Vietnamese => "vi",
            Language::Chinese => "zh",
            Language::Bengali => "bn",
            Language::Gujarati => "gu",
            Language::Hindi => "hi",
            Language::Kannada => "kn",
            Language::Marathi => "mr",
            Language::Nepali => "ne",
            Language::Oriya => "or",
            Language::Panjabi => "pa",
            Language::Sindhi => "sd",
            Language::Sinhala => "si",
            Language::Urdu => "ur",
            Language::Tamil => "ta",
            Language::Cebuano => "ceb",
            Language::Iloko => "ilo",
            Language::Indonesian => "id",
            Language::Javanese => "jv",
            Language::Malagasy => "mg",
            Language::PlateauMalagasy => return None,
            Language::Malay => "zsm_Latn",
            Language::Malayalam => "ml",
            Language::Sundanese => "su",
            Language::Tagalog => "tl",
            Language::Burmese => "my",
            Language::Khmer => "km",
            Language::Lao => "lo",
            Language::Thai => "th",
            Language::Mongolian => "mn",
            Language::NajdiArabic => "ar",
            Language::Hebrew => "he",
            Language::SouthernPashto => "ps",
            Language::Pashto => "ps",
            Language::Farsi => "fa",
            Language::Faroese => "fo",
            Language::Amharic => "am",
            Language::Fulah => "ff",
            Language::Hausa => "ha",
            Language::Igbo => "ig",
            Language::Lingala => "ln",
            Language::Luganda => "lg",
            Language::NorthernSotho => "nso",
            Language::Somali => "so",
            Language::Swahili => "sw",
            Language::Swati => "ss",
            Language::Tswana => "tn",
            Language::Wolof => "wo",
            Language::Xhosa => "xh",
            Language::Yoruba => "yo",
            Language::Zulu => "zu",
            Language::Haitian => "ht",
            Language::Latvian => "lv",
            Language::Achinese => return None,
            Language::MesopotamianArabic => return None,
            Language::TaizziAdeniArabic => return None,
            Language::TunisianArabic => return None,
            Language::SouthLevantineArabic => return None,
            Language::Akan => "ak",
            Language::NorthLevantineArabic => return None,
            Language::MoroccanArabic => return None,
            Language::EgyptianArabic => return None,
            Language::Assamese => "as",
            Language::Awadhi => return None,
            Language::CentralAymara => return None,
            Language::SouthAzerbaijani => return None,
            Language::Bambara => "bm",
            Language::Balinese => return None,
            Language::Bemba => return None,
            Language::Bhojpuri => return None,
            Language::Banjar => return None,
            Language::Tibetan => "bo",
            Language::Buginese => return None,
            Language::Chokwe => return None,
            Language::CentralKurdish => return None,
            Language::CrimeanTatar => return None,
            Language::SouthwesternDinka => return None,
            Language::Dyula => return None,
            Language::Dzongkha => "dz",
            Language::Esperanto => "eo",
            Language::Basque => "eu",
            Language::Ewe => "ee",
            Language::Fijian => "fi",
            Language::Fon => return None,
            Language::Friulian => return None,
            Language::NigerianFulfulde => return None,
            Language::WestCentralOromo => return None,
            Language::Guarani => "gn",
            Language::Chhattisgarhi => return None,
            Language::Kabyle => return None,
            Language::Kachin => return None,
            Language::Kamba => return None,
            Language::Kashmiri => "ks",
            Language::Kabiye => return None,
            Language::Kabuverdianu => return None,
            Language::HalhMongolian => return None,
            Language::Kikuyu => "ki",
            Language::Kinyarwanda => "rw",
            Language::Kirghiz => "ky",
            Language::Kimbundu => return None,
            Language::NorthernKurdish => return None,
            Language::CentralKanuri => return None,
            Language::Kongo => "kg",
            Language::Ligurian => return None,
            Language::Limburgan => "li",
            Language::Lombard => return None,
            Language::Latgalian => return None,
            Language::LubaLulua => return None,
            Language::Ganda => "lg",
            Language::Luo => return None,
            Language::Lushai => return None,
            Language::Magahi => return None,
            Language::Maithili => return None,
            Language::Minangkabau => return None,
            Language::Maltese => "mt",
            Language::Manipuri => return None,
            Language::Mossi => return None,
            Language::Maori => "mi",
            Language::NorwegianNynorsk => "nn",
            Language::NorwegianBokmal => "nb",
            Language::Pedi => return None,
            Language::Nuer => return None,
            Language::Nyanja => "ny",
            Language::Odia => return None,
            Language::Pangasinan => return None,
            Language::Papiamento => return None,
            Language::IranianPersian => return None,
            Language::Dari => return None,
            Language::AyacuchoQuechua => return None,
            Language::Rundi => "rn",
            Language::Sango => "sg",
            Language::Sanskrit => return None,
            Language::Santali => return None,
            Language::Sicilian => return None,
            Language::Shan => return None,
            Language::Samoan => "sm",
            Language::Shona => "sn",
            Language::SouthernSotho => "st",
            Language::Sardinian => "sc",
            Language::Silesian => return None,
            Language::Tamasheq => return None,
            Language::Tatar => "tt",
            Language::Telugu => "te",
            Language::Tajik => "tg",
            Language::Tigrinya => "ti",
            Language::TokPisin => return None,
            Language::Tsonga => "ts",
            Language::Turkmen => "tk",
            Language::Tumbuka => return None,
            Language::Twi => "tw",
            Language::CentralAtlasTamazight => return None,
            Language::Uighur => "ug",
            Language::Umbundu => return None,
            Language::Venetian => return None,
            Language::Waray => return None,
            Language::EasternYiddish => return None,
            Language::YueChinese => return None,
            Language::Arabic => "ar",
            Language::Albanian => "sq",
            Language::Azerbaijani => "az",
            Language::ChineseMandarin => return None,
            Language::HaitianCreole => "ht",
            Language::CentralKhmer => "km",
        };

        Some(code)
    }

    pub fn get_nllb_code(&self) -> Option<&'static str> {
        let result = match self {
            Self::Nepali
            | Self::ChineseMandarin
            | Self::Breton
            | Self::Norwegian
            | Self::Malagasy
            | Self::Azerbaijani
            | Self::WesternFrisian
            | Self::Pashto
            | Self::Farsi
            | Self::Fulah
            | Self::Mongolian
            | Self::Yiddish => return None,

            Language::Afrikaans => "afr_Latn",
            Language::Danish => "dan_Latn",
            Language::Dutch => "nld_Latn",
            Language::German => "deu_Latn",
            Language::English => "eng_Latn",
            Language::Icelandic => "isl_Latn",
            Language::Luxembourgish => "ltz_Latn",
            Language::Swedish => "swe_Latn",
            Language::Asturian => "ast_Latn",
            Language::Catalan => "cat_Latn",
            Language::French => "fra_Latn",
            Language::Galician => "glg_Latn",
            Language::Italian => "ita_Latn",
            Language::Occitan => "oci_Latn",
            Language::Portuguese => "por_Latn",
            Language::Romanian => "ron_Latn",
            Language::Spanish => "spa_Latn",
            Language::Belarusian => "bel_Cyrl",
            Language::Bosnian => "bos_Latn",
            Language::Bulgarian => "bul_Cyrl",
            Language::Croatian => "hrv_Latn",
            Language::Czech => "ces_Latn",
            Language::Macedonian => "mkd_Cyrl",
            Language::Polish => "pol_Latn",
            Language::Russian => "rus_Cyrl",
            Language::Serbian => "srp_Cyrl",
            Language::Slovak => "slk_Latn",
            Language::Slovenian => "slv_Latn",
            Language::Ukrainian => "ukr_Cyrl",
            Language::Estonian => "est_Latn",
            Language::Finnish => "fin_Latn",
            Language::Hungarian => "hun_Latn",
            Language::Latvian => "lvs_Latn",
            Language::Lithuanian => "lit_Latn",
            Language::Albanian => "als_Latn",
            Language::Armenian => "hye_Armn",
            Language::Georgian => "kat_Geor",
            Language::Greek => "ell_Grek",
            Language::Irish => "gle_Latn",
            Language::ScottishGaelic => "gla_Latn",
            Language::Welsh => "cym_Latn",
            Language::Bashkir => "bak_Cyrl",
            Language::Kazakh => "kaz_Cyrl",
            Language::Turkish => "tur_Latn",
            Language::Uzbek => "uzn_Latn",
            Language::Japanese => "jpn_Jpan",
            Language::Korean => "kor_Hang",
            Language::Vietnamese => "vie_Latn",
            Language::Bengali => "ben_Beng",
            Language::Gujarati => "guj_Gujr",
            Language::Hindi => "hin_Deva",
            Language::Kannada => "kan_Knda",
            Language::Marathi => "mar_Deva",
            Language::Oriya => "ory_Orya",
            Language::Panjabi => "pan_Guru",
            Language::Sindhi => "snd_Arab",
            Language::Sinhala => "sin_Sinh",
            Language::Urdu => "urd_Arab",
            Language::Tamil => "tam_Taml",
            Language::Cebuano => "ceb_Latn",
            Language::Iloko => "ilo_Latn",
            Language::Indonesian => "ind_Latn",
            Language::Javanese => "jav_Latn",
            Language::Malay => "zsm_Latn",
            Language::Malayalam => "mal_Mlym",
            Language::Sundanese => "sun_Latn",
            Language::Tagalog => "tgl_Latn",
            Language::Burmese => "mya_Mymr",
            Language::CentralKhmer => "khm_Khmr",
            Language::Lao => "lao_Laoo",
            Language::Thai => "tha_Thai",
            Language::Hebrew => "heb_Hebr",
            Language::Amharic => "amh_Ethi",
            Language::Hausa => "hau_Latn",
            Language::Igbo => "ibo_Latn",
            Language::Lingala => "lin_Latn",
            Language::Luganda => "lug_Latn",
            Language::NorthernSotho => "nso_Latn",
            Language::Somali => "som_Latn",
            Language::Swahili => "swh_Latn",
            Language::Swati => "ssw_Latn",
            Language::Tswana => "tsn_Latn",
            Language::Wolof => "wol_Latn",
            Language::Xhosa => "xho_Latn",
            Language::Yoruba => "yor_Latn",
            Language::Zulu => "zul_Latn",
            Language::HaitianCreole => "hat_Latn",
            Language::Achinese => "ace_Arab",
            Language::MesopotamianArabic => "acm_Arab",
            Language::TaizziAdeniArabic => "acq_Arab",
            Language::TunisianArabic => "aeb_Arab",
            Language::SouthLevantineArabic => "ajp_Arab",
            Language::Akan => "aka_Latn",
            Language::NorthLevantineArabic => "apc_Arab",
            Language::Arabic => "arb_Arab",
            Language::NajdiArabic => "ars_Arab",
            Language::MoroccanArabic => "ary_Arab",
            Language::EgyptianArabic => "arz_Arab",
            Language::Assamese => "asm_Beng",
            Language::Awadhi => "awa_Deva",
            Language::CentralAymara => "ayr_Latn",
            Language::SouthAzerbaijani => "azb_Arab",
            Language::NorthAzerbaijani => "azj_Latn",
            Language::Bambara => "bam_Latn",
            Language::Balinese => "ban_Latn",
            Language::Bemba => "bem_Latn",
            Language::Bhojpuri => "bho_Deva",
            Language::Banjar => "bjn_Arab",
            Language::Tibetan => "bod_Tibt",
            Language::Buginese => "bug_Latn",
            Language::Chokwe => "cjk_Latn",
            Language::CentralKurdish => "ckb_Arab",
            Language::CrimeanTatar => "crh_Latn",
            Language::SouthwesternDinka => "dik_Latn",
            Language::Dyula => "dyu_Latn",
            Language::Dzongkha => "dzo_Tibt",
            Language::Esperanto => "epo_Latn",
            Language::Basque => "eus_Latn",
            Language::Ewe => "ewe_Latn",
            Language::Faroese => "fao_Latn",
            Language::Fijian => "fij_Latn",
            Language::Fon => "fon_Latn",
            Language::Friulian => "fur_Latn",
            Language::NigerianFulfulde => "fuv_Latn",
            Language::WestCentralOromo => "gaz_Latn",
            Language::Guarani => "grn_Latn",
            Language::Haitian => "hat_Latn",
            Language::Chhattisgarhi => "hne_Deva",
            Language::Kabyle => "kab_Latn",
            Language::Kachin => "kac_Latn",
            Language::Kamba => "kam_Latn",
            Language::Kashmiri => "kas_Arab",
            Language::Kabiye => "kbp_Latn",
            Language::Kabuverdianu => "kea_Latn",
            Language::HalhMongolian => "khk_Cyrl",
            Language::Khmer => "khm_Khmr",
            Language::Kikuyu => "kik_Latn",
            Language::Kinyarwanda => "kin_Latn",
            Language::Kirghiz => "kir_Cyrl",
            Language::Kimbundu => "kmb_Latn",
            Language::NorthernKurdish => "kmr_Latn",
            Language::CentralKanuri => "knc_Latn",
            Language::Kongo => "kon_Latn",
            Language::Ligurian => "lij_Latn",
            Language::Limburgan => "lim_Latn",
            Language::Lombard => "lmo_Latn",
            Language::Latgalian => "ltg_Latn",
            Language::LubaLulua => "lua_Latn",
            Language::Ganda => "lug_Latn",
            Language::Luo => "luo_Latn",
            Language::Lushai => "lus_Latn",
            Language::Magahi => "mag_Deva",
            Language::Maithili => "mai_Deva",
            Language::Minangkabau => "min_Latn",
            Language::Maltese => "mlt_Latn",
            Language::Manipuri => "mni_Beng",
            Language::Mossi => "mos_Latn",
            Language::Maori => "mri_Latn",
            Language::NorwegianNynorsk => "nno_Latn",
            Language::NorwegianBokmal => "nob_Latn",
            Language::Pedi => "nso_Latn",
            Language::Nuer => "nus_Latn",
            Language::Nyanja => "nya_Latn",
            Language::Odia => "ory_Orya",
            Language::Pangasinan => "pag_Latn",
            Language::Papiamento => "pap_Latn",
            Language::SouthernPashto => "pbt_Arab",
            Language::IranianPersian => "pes_Arab",
            Language::PlateauMalagasy => "plt_Latn",
            Language::Dari => "prs_Arab",
            Language::AyacuchoQuechua => "quy_Latn",
            Language::Rundi => "run_Latn",
            Language::Sango => "sag_Latn",
            Language::Sanskrit => "san_Deva",
            Language::Santali => "sat_Beng",
            Language::Sicilian => "scn_Latn",
            Language::Shan => "shn_Mymr",
            Language::Samoan => "smo_Latn",
            Language::Shona => "sna_Latn",
            Language::SouthernSotho => "sot_Latn",
            Language::Sardinian => "srd_Latn",
            Language::Silesian => "szl_Latn",
            Language::Tamasheq => "taq_Latn",
            Language::Tatar => "tat_Cyrl",
            Language::Telugu => "tel_Telu",
            Language::Tajik => "tgk_Cyrl",
            Language::Tigrinya => "tir_Ethi",
            Language::TokPisin => "tpi_Latn",
            Language::Tsonga => "tso_Latn",
            Language::Turkmen => "tuk_Latn",
            Language::Tumbuka => "tum_Latn",
            Language::Twi => "twi_Latn",
            Language::CentralAtlasTamazight => "tzm_Tfng",
            Language::Uighur => "uig_Arab",
            Language::Umbundu => "umb_Latn",
            Language::NorthernUzbek => "uzn_Latn",
            Language::Venetian => "vec_Latn",
            Language::Waray => "war_Latn",
            Language::EasternYiddish => "ydd_Hebr",
            Language::YueChinese => "yue_Hant",
            Language::Chinese => "zho_Hans",
        };
        Some(result)
    }

    pub fn get_iso_639_3_code(&self) -> &'static str {
        match self {
            Language::Afrikaans => "afr",
            Language::Danish => "dan",
            Language::Dutch => "nld",
            Language::German => "deu",
            Language::English => "eng",
            Language::Icelandic => "isl",
            Language::Luxembourgish => "ltz",
            Language::Norwegian => "nor",
            Language::Swedish => "swe",
            Language::WesternFrisian => "fry",
            Language::Yiddish => "yid",
            Language::Asturian => "ast",
            Language::Catalan => "cat",
            Language::French => "fra",
            Language::Galician => "glg",
            Language::Italian => "ita",
            Language::Occitan => "oci",
            Language::Portuguese => "por",
            Language::Romanian => "ron",
            Language::Spanish => "spa",
            Language::Belarusian => "bel",
            Language::Bosnian => "bos",
            Language::Bulgarian => "bul",
            Language::Croatian => "hrv",
            Language::Czech => "ces",
            Language::Macedonian => "mkd",
            Language::Polish => "pol",
            Language::Russian => "rus",
            Language::Serbian => "srp",
            Language::Slovak => "slk",
            Language::Slovenian => "slv",
            Language::Ukrainian => "ukr",
            Language::Estonian => "est",
            Language::Finnish => "fin",
            Language::Hungarian => "hun",
            Language::Latvian => "lav",
            Language::Lithuanian => "lit",
            Language::Albanian => "sqi",
            Language::Armenian => "hye",
            Language::Georgian => "kat",
            Language::Greek => "ell",
            Language::Breton => "bre",
            Language::Irish => "gle",
            Language::ScottishGaelic => "gla",
            Language::Welsh => "cym",
            Language::Azerbaijani => "aze",
            Language::Bashkir => "bak",
            Language::Kazakh => "kaz",
            Language::Turkish => "tur",
            Language::Uzbek => "uzb",
            Language::Japanese => "jpn",
            Language::Korean => "kor",
            Language::Vietnamese => "vie",
            Language::ChineseMandarin => "cmn",
            Language::Bengali => "ben",
            Language::Gujarati => "guj",
            Language::Hindi => "hin",
            Language::Kannada => "kan",
            Language::Marathi => "mar",
            Language::Nepali => "nep",
            Language::Oriya => "ori",
            Language::Panjabi => "pan",
            Language::Sindhi => "snd",
            Language::Sinhala => "sin",
            Language::Urdu => "urd",
            Language::Tamil => "tam",
            Language::Cebuano => "ceb",
            Language::Iloko => "ilo",
            Language::Indonesian => "ind",
            Language::Javanese => "jav",
            Language::Malagasy => "mlg",
            Language::Malay => "msa",
            Language::Malayalam => "mal",
            Language::Sundanese => "sun",
            Language::Tagalog => "tgl",
            Language::Burmese => "mya",
            Language::CentralKhmer => "khm",
            Language::Lao => "lao",
            Language::Thai => "tha",
            Language::Mongolian => "mon",
            Language::Arabic => "ara",
            Language::Hebrew => "heb",
            Language::Pashto => "pus",
            Language::Farsi => "fas",
            Language::Amharic => "amh",
            Language::Fulah => "ful",
            Language::Hausa => "hau",
            Language::Igbo => "ibo",
            Language::Lingala => "lin",
            Language::Luganda => "lug",
            Language::NorthernSotho => "nso",
            Language::Somali => "som",
            Language::Swahili => "swa",
            Language::Swati => "ssw",
            Language::Tswana => "tsn",
            Language::Wolof => "wol",
            Language::Xhosa => "xho",
            Language::Yoruba => "yor",
            Language::Zulu => "zul",
            Language::HaitianCreole => "hat",
            Language::Achinese => "ace",
            Language::MesopotamianArabic => "acm",
            Language::TaizziAdeniArabic => "acq",
            Language::TunisianArabic => "aeb",
            Language::SouthLevantineArabic => "ajp",
            Language::Akan => "aka",
            Language::NorthLevantineArabic => "apc",
            Language::NajdiArabic => "ars",
            Language::MoroccanArabic => "ary",
            Language::EgyptianArabic => "arz",
            Language::Assamese => "asm",
            Language::Awadhi => "awa",
            Language::CentralAymara => "ayr",
            Language::SouthAzerbaijani => "azb",
            Language::NorthAzerbaijani => "azj",
            Language::Bambara => "bam",
            Language::Balinese => "ban",
            Language::Bemba => "bem",
            Language::Bhojpuri => "bho",
            Language::Banjar => "bjn",
            Language::Tibetan => "bod",
            Language::Buginese => "bug",
            Language::Chokwe => "cjk",
            Language::CentralKurdish => "ckb",
            Language::CrimeanTatar => "crh",
            Language::SouthwesternDinka => "dik",
            Language::Dyula => "dyu",
            Language::Dzongkha => "dzo",
            Language::Esperanto => "epo",
            Language::Basque => "eus",
            Language::Ewe => "ewe",
            Language::Faroese => "fao",
            Language::Fijian => "fij",
            Language::Fon => "fon",
            Language::Friulian => "fur",
            Language::NigerianFulfulde => "fuv",
            Language::WestCentralOromo => "gaz",
            Language::Guarani => "grn",
            Language::Haitian => "hat",
            Language::Chhattisgarhi => "hne",
            Language::Kabyle => "kab",
            Language::Kachin => "kac",
            Language::Kamba => "kam",
            Language::Kashmiri => "kas",
            Language::Kabiye => "kbp",
            Language::Kabuverdianu => "kea",
            Language::HalhMongolian => "khk",
            Language::Khmer => "khm",
            Language::Kikuyu => "kik",
            Language::Kinyarwanda => "kin",
            Language::Kirghiz => "kir",
            Language::Kimbundu => "kmb",
            Language::NorthernKurdish => "kmr",
            Language::CentralKanuri => "knc",
            Language::Kongo => "kon",
            Language::Ligurian => "lij",
            Language::Limburgan => "lim",
            Language::Lombard => "lmo",
            Language::Latgalian => "ltg",
            Language::LubaLulua => "lua",
            Language::Ganda => "lug",
            Language::Luo => "luo",
            Language::Lushai => "lus",
            Language::Magahi => "mag",
            Language::Maithili => "mai",
            Language::Minangkabau => "min",
            Language::Maltese => "mlt",
            Language::Manipuri => "mni",
            Language::Mossi => "mos",
            Language::Maori => "mri",
            Language::NorwegianNynorsk => "nno",
            Language::NorwegianBokmal => "nob",
            Language::Pedi => "nso",
            Language::Nuer => "nus",
            Language::Nyanja => "nya",
            Language::Odia => "ory",
            Language::Pangasinan => "pag",
            Language::Papiamento => "pap",
            Language::SouthernPashto => "pbt",
            Language::IranianPersian => "pes",
            Language::PlateauMalagasy => "plt",
            Language::Dari => "prs",
            Language::AyacuchoQuechua => "quy",
            Language::Rundi => "run",
            Language::Sango => "sag",
            Language::Sanskrit => "san",
            Language::Santali => "sat",
            Language::Sicilian => "scn",
            Language::Shan => "shn",
            Language::Samoan => "smo",
            Language::Shona => "sna",
            Language::SouthernSotho => "sot",
            Language::Sardinian => "srd",
            Language::Silesian => "szl",
            Language::Tamasheq => "taq",
            Language::Tatar => "tat",
            Language::Telugu => "tel",
            Language::Tajik => "tgk",
            Language::Tigrinya => "tir",
            Language::TokPisin => "tpi",
            Language::Tsonga => "tso",
            Language::Turkmen => "tuk",
            Language::Tumbuka => "tum",
            Language::Twi => "twi",
            Language::CentralAtlasTamazight => "tzm",
            Language::Uighur => "uig",
            Language::Umbundu => "umb",
            Language::NorthernUzbek => "uzn",
            Language::Venetian => "vec",
            Language::Waray => "war",
            Language::EasternYiddish => "ydd",
            Language::YueChinese => "yue",
            Language::Chinese => "zho",
        }
    }
}

/// # Configuration for text translation
/// Contains information regarding the model to load, mirrors the GenerationConfig, with a
/// different set of default parameters and sets the device to place the model on.
pub struct TranslationConfig {
    /// Model type used for translation
    pub model_type: ModelType,
    /// Model weights resource
    pub model_resource: Box<dyn ResourceProvider + Send>,
    /// Config resource
    pub config_resource: Box<dyn ResourceProvider + Send>,
    /// Vocab resource
    pub vocab_resource: Box<dyn ResourceProvider + Send>,
    /// Merges resource
    pub merges_resource: Option<Box<dyn ResourceProvider + Send>>,
    /// Supported source languages
    pub source_languages: HashSet<Language>,
    /// Supported target languages
    pub target_languages: HashSet<Language>,
    /// Minimum sequence length (default: 0)
    pub min_length: i64,
    /// Maximum sequence length (default: 512)
    pub max_length: Option<i64>,
    /// Sampling flag. If true, will perform top-k and/or nucleus sampling on generated tokens, otherwise greedy (deterministic) decoding (default: true)
    pub do_sample: bool,
    /// Early stopping flag indicating if the beam search should stop as soon as `num_beam` hypotheses have been generated (default: false)
    pub early_stopping: bool,
    /// Number of beams for beam search (default: 5)
    pub num_beams: i64,
    /// Temperature setting. Values higher than 1 will improve originality at the risk of reducing relevance (default: 1.0)
    pub temperature: f64,
    /// Top_k values for sampling tokens. Value higher than 0 will enable the feature (default: 0)
    pub top_k: i64,
    /// Top_p value for [Nucleus sampling, Holtzman et al.](http://arxiv.org/abs/1904.09751). Keep top tokens until cumulative probability reaches top_p (default: 0.9)
    pub top_p: f64,
    /// Repetition penalty (mostly useful for CTRL decoders). Values higher than 1 will penalize tokens that have been already generated. (default: 1.0)
    pub repetition_penalty: f64,
    /// Exponential penalty based on the length of the hypotheses generated (default: 1.0)
    pub length_penalty: f64,
    /// Number of allowed repetitions of n-grams. Values higher than 0 turn on this feature (default: 3)
    pub no_repeat_ngram_size: i64,
    /// Number of sequences to return for each prompt text (default: 1)
    pub num_return_sequences: i64,
    /// Device to place the model on (default: CUDA/GPU when available)
    pub device: Device,
    /// Number of beam groups for diverse beam generation. If provided and higher than 1, will split the beams into beam subgroups leading to more diverse generation.
    pub num_beam_groups: Option<i64>,
    /// Diversity penalty for diverse beam search. High values will enforce more difference between beam groups (default: 5.5)
    pub diversity_penalty: Option<f64>,
}

impl TranslationConfig {
    /// Create a new `TranslationConfiguration` from an available language.
    ///
    /// # Arguments
    ///
    /// * `language` - `Language` enum value (e.g. `Language::EnglishToFrench`)
    /// * `device` - `Device` to place the model on (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {     ///
    /// use rust_bert::marian::{
    ///     MarianConfigResources, MarianModelResources, MarianSourceLanguages, MarianSpmResources,
    ///     MarianTargetLanguages, MarianVocabResources,
    /// };
    /// use rust_bert::pipelines::common::ModelType;
    /// use rust_bert::pipelines::translation::TranslationConfig;
    /// use rust_bert::resources::RemoteResource;
    /// use tch::Device;
    ///
    /// let model_resource = RemoteResource::from_pretrained(MarianModelResources::ROMANCE2ENGLISH);
    /// let config_resource = RemoteResource::from_pretrained(MarianConfigResources::ROMANCE2ENGLISH);
    /// let vocab_resource = RemoteResource::from_pretrained(MarianVocabResources::ROMANCE2ENGLISH);
    /// let spm_resource = RemoteResource::from_pretrained(MarianSpmResources::ROMANCE2ENGLISH);
    ///
    /// let source_languages = MarianSourceLanguages::ROMANCE2ENGLISH;
    /// let target_languages = MarianTargetLanguages::ROMANCE2ENGLISH;
    ///
    /// let translation_config = TranslationConfig::new(
    ///     ModelType::Marian,
    ///     model_resource,
    ///     config_resource,
    ///     vocab_resource,
    ///     Some(spm_resource),
    ///     source_languages,
    ///     target_languages,
    ///     Device::cuda_if_available(),
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn new<RM, RC, RV, S, T>(
        model_type: ModelType,
        model_resource: RM,
        config_resource: RC,
        vocab_resource: RV,
        merges_resource: Option<RV>,
        source_languages: S,
        target_languages: T,
        device: impl Into<Option<Device>>,
    ) -> TranslationConfig
    where
        RM: ResourceProvider + Send + 'static,
        RC: ResourceProvider + Send + 'static,
        RV: ResourceProvider + Send + 'static,
        S: AsRef<[Language]>,
        T: AsRef<[Language]>,
    {
        let device = device.into().unwrap_or_else(Device::cuda_if_available);

        TranslationConfig {
            model_type,
            model_resource: Box::new(model_resource),
            config_resource: Box::new(config_resource),
            vocab_resource: Box::new(vocab_resource),
            merges_resource: merges_resource.map(|r| Box::new(r) as Box<_>),
            source_languages: source_languages.as_ref().iter().cloned().collect(),
            target_languages: target_languages.as_ref().iter().cloned().collect(),
            device,
            min_length: 0,
            max_length: Some(512),
            do_sample: false,
            early_stopping: true,
            num_beams: 3,
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            no_repeat_ngram_size: 0,
            num_return_sequences: 1,
            num_beam_groups: None,
            diversity_penalty: None,
        }
    }
}

impl From<TranslationConfig> for GenerateConfig {
    fn from(config: TranslationConfig) -> GenerateConfig {
        GenerateConfig {
            model_resource: config.model_resource,
            config_resource: config.config_resource,
            merges_resource: config.merges_resource,
            vocab_resource: config.vocab_resource,
            min_length: config.min_length,
            max_length: config.max_length,
            do_sample: config.do_sample,
            early_stopping: config.early_stopping,
            num_beams: config.num_beams,
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repetition_penalty: config.repetition_penalty,
            length_penalty: config.length_penalty,
            no_repeat_ngram_size: config.no_repeat_ngram_size,
            num_return_sequences: config.num_return_sequences,
            num_beam_groups: config.num_beam_groups,
            diversity_penalty: config.diversity_penalty,
            device: config.device,
        }
    }
}

#[allow(clippy::large_enum_variant)]
/// # Abstraction that holds one particular translation model, for any of the supported models
pub enum TranslationOption {
    /// Translator based on Marian model
    Marian(MarianGenerator),
    /// Translator based on T5 model
    T5(T5Generator),
    /// Translator based on MBart50 model
    MBart(MBartGenerator),
    /// Translator based on M2M100 model
    M2M100(M2M100Generator),
    NLLB(NLLBGenerator),
}

impl TranslationOption {
    pub fn new(config: TranslationConfig) -> Result<Self, RustBertError> {
        match config.model_type {
            ModelType::Marian => Ok(TranslationOption::Marian(MarianGenerator::new(
                config.into(),
            )?)),
            ModelType::T5 => Ok(TranslationOption::T5(T5Generator::new(config.into())?)),
            ModelType::MBart => Ok(TranslationOption::MBart(MBartGenerator::new(
                config.into(),
            )?)),
            ModelType::M2M100 => Ok(TranslationOption::M2M100(M2M100Generator::new(
                config.into(),
            )?)),
            ModelType::NLLB => {
                let config: GenerateConfig = config.into();
                let tokenizer = TokenizerOption::from_file(
                    ModelType::NLLB,
                    config.vocab_resource.get_local_path()?.to_str().unwrap(),
                    Some(
                        config
                            .merges_resource
                            .as_ref()
                            .ok_or_else(|| {
                                RustBertError::InvalidConfigurationError(
                                    "M2M100 expects a merges resources to be provided".to_string(),
                                )
                            })?
                            .get_local_path()?
                            .to_str()
                            .unwrap(),
                    ),
                    false,
                    None,
                    None,
                )?;

                Ok(TranslationOption::NLLB(NLLBGenerator::new_with_tokenizer(
                    config, tokenizer,
                )?))
            }
            _ => Err(RustBertError::InvalidConfigurationError(format!(
                "Translation not implemented for {:?}!",
                config.model_type
            ))),
        }
    }

    pub fn new_with_tokenizer(
        config: TranslationConfig,
        tokenizer: TokenizerOption,
    ) -> Result<Self, RustBertError> {
        match config.model_type {
            ModelType::Marian => Ok(TranslationOption::Marian(
                MarianGenerator::new_with_tokenizer(config.into(), tokenizer)?,
            )),
            ModelType::T5 => Ok(TranslationOption::T5(T5Generator::new_with_tokenizer(
                config.into(),
                tokenizer,
            )?)),
            ModelType::MBart => Ok(TranslationOption::MBart(
                MBartGenerator::new_with_tokenizer(config.into(), tokenizer)?,
            )),
            ModelType::M2M100 => Ok(TranslationOption::M2M100(
                M2M100Generator::new_with_tokenizer(config.into(), tokenizer)?,
            )),
            ModelType::NLLB => Ok(TranslationOption::NLLB(NLLBGenerator::new_with_tokenizer(
                config.into(),
                tokenizer,
            )?)),
            _ => Err(RustBertError::InvalidConfigurationError(format!(
                "Translation not implemented for {:?}!",
                config.model_type
            ))),
        }
    }

    /// Returns the `ModelType` for this TranslationOption
    pub fn model_type(&self) -> ModelType {
        match *self {
            Self::Marian(_) => ModelType::Marian,
            Self::T5(_) => ModelType::T5,
            Self::MBart(_) => ModelType::MBart,
            Self::M2M100(_) => ModelType::M2M100,
            Self::NLLB(_) => ModelType::NLLB,
        }
    }

    fn validate_and_get_prefix_and_forced_bos_id(
        &self,
        source_language: Option<&Language>,
        target_language: Option<&Language>,
        supported_source_languages: &HashSet<Language>,
        supported_target_languages: &HashSet<Language>,
    ) -> Result<(Option<String>, Option<i64>), RustBertError> {
        if let Some(source_language) = source_language {
            if !supported_source_languages.contains(source_language) {
                return Err(RustBertError::ValueError(format!(
                    "{source_language} not in list of supported languages: {supported_source_languages:?}",
                )));
            }
        }

        if let Some(target_language) = target_language {
            if !supported_target_languages.contains(target_language) {
                return Err(RustBertError::ValueError(format!(
                    "{target_language} not in list of supported languages: {supported_target_languages:?}"
                )));
            }
        }

        Ok(match *self {
            Self::Marian(_) => {
                if supported_target_languages.len() > 1 {
                    (
                        Some(format!(
                            ">>{}<< ",
                            target_language.and_then(|l| l.get_iso_639_1_code()).ok_or_else(|| RustBertError::ValueError(format!(
                                        "Missing target language for Marian \
                                        (multiple languages supported by model: {supported_target_languages:?}, \
                                        need to specify target language)",
                                    )))?
                        )),
                        None,
                    )
                } else {
                    (None, None)
                }
            }
            Self::T5(_) => (
                Some(format!(
                    "translate {} to {}:",
                    source_language.ok_or_else(|| RustBertError::ValueError(
                                "Missing source language for T5".to_string(),
                            ))?,
                    target_language.ok_or_else(|| RustBertError::ValueError(
                                "Missing target language for T5".to_string(),
                            ))?,
                )),
                None,
            ),
            Self::MBart(ref model) => {
                (
                    Some(format!(
                        ">>{}<< ",
                        source_language.and_then(|l| l.get_iso_639_1_code()).ok_or_else(|| RustBertError::ValueError(format!(
                                "Missing source language for MBart\
                                (multiple languages supported by model: {supported_source_languages:?}, \
                                need to specify target language)"
                            )))?
                    )),
                    if let Some(target_language) = target_language {
                        Some(
                        model._get_tokenizer().convert_tokens_to_ids(&[format!(
                            ">>{}<<",
                            target_language.get_iso_639_1_code().ok_or_else(|| {
                                RustBertError::ValueError(format!(
                                    "This language has no ISO639-I code. Languages supported by model: {supported_source_languages:?}."
                                ))
                            })?
                        )])[0],
                    )
                    } else {
                        return Err(RustBertError::ValueError(format!(
                            "Missing target language for MBart\
                        (multiple languages supported by model: {supported_target_languages:?}, \
                        need to specify target language)"
                        )));
                    },
                )
            }
            Self::M2M100(ref model) => (
                Some(match source_language {
                    Some(value) => {
                        let language_code = value.get_iso_639_1_code().ok_or_else(|| {
                            RustBertError::ValueError(format!(
                                "This language has no ISO639-I language code representation. \
                                languages supported by the model: {supported_target_languages:?}"
                            ))
                        })?;
                        match language_code.len() {
                            2 => format!(">>{language_code}.<< "),
                            3 => format!(">>{language_code}<< "),
                            _ => {
                                return Err(RustBertError::ValueError(
                                    "Invalid ISO 639-I code".to_string(),
                                ));
                            }
                        }
                    }
                    None => {
                        return Err(RustBertError::ValueError(format!(
                            "Missing source language for M2M100 \
                            (multiple languages supported by model: {supported_source_languages:?}, \
                            need to specify target language)"
                        )));
                    }
                }),
                if let Some(target_language) = target_language {
                    let language_code = target_language.get_iso_639_1_code().ok_or_else(|| {
                        RustBertError::ValueError(format!(
                            "This language has no ISO639-I language code representation. \
                            languages supported by the model: {supported_target_languages:?}"
                        ))
                    })?;
                    Some(
                        model._get_tokenizer().convert_tokens_to_ids(&[
                            match language_code.len() {
                                2 => format!(">>{language_code}.<<"),
                                3 => format!(">>{language_code}<<"),
                                _ => {
                                    return Err(RustBertError::ValueError(
                                        "Invalid ISO 639-3 code".to_string(),
                                    ));
                                }
                            },
                        ])[0],
                    )
                } else {
                    return Err(RustBertError::ValueError(format!(
                        "Missing target language for M2M100 \
                        (multiple languages supported by model: {supported_target_languages:?}, \
                        need to specify target language)",
                    )));
                },
            ),
            Self::NLLB(ref model) => {
                let source_language = source_language
                    .and_then(Language::get_nllb_code)
                    .map(str::to_string)
                    .ok_or_else(|| RustBertError::ValueError(
                        format!("Missing source language for NLLB. Need to specify one from: {supported_source_languages:?}")
                ))?;

                let target_language = target_language
                    .and_then(Language::get_nllb_code)
                    .map(str::to_string)
                    .map(|code| model._get_tokenizer().convert_tokens_to_ids(&[code])[0])
                    .ok_or_else(|| RustBertError::ValueError(
                        format!("Missing target language for NLLB. Need to specify one from: {supported_target_languages:?}")
                ))?;

                (Some(source_language), Some(target_language))
            }
        })
    }

    /// Interface method to generate() of the particular models.
    pub fn generate<S>(
        &self,
        prompt_texts: Option<&[S]>,
        forced_bos_token_id: Option<i64>,
    ) -> Vec<String>
    where
        S: AsRef<str> + Sync,
    {
        match *self {
            Self::Marian(ref model) => model
                .generate(prompt_texts, None)
                .into_iter()
                .map(|output| output.text)
                .collect(),
            Self::T5(ref model) => model
                .generate(prompt_texts, None)
                .into_iter()
                .map(|output| output.text)
                .collect(),
            Self::MBart(ref model) => {
                let generate_options = GenerateOptions {
                    forced_bos_token_id,
                    ..Default::default()
                };
                model
                    .generate(prompt_texts, Some(generate_options))
                    .into_iter()
                    .map(|output| output.text)
                    .collect()
            }
            Self::M2M100(ref model) | Self::NLLB(ref model) => {
                let generate_options = GenerateOptions {
                    forced_bos_token_id,
                    ..Default::default()
                };
                model
                    .generate(prompt_texts, Some(generate_options))
                    .into_iter()
                    .map(|output| output.text)
                    .collect()
            }
        }
    }
}

/// # TranslationModel to perform translation
pub struct TranslationModel {
    model: TranslationOption,
    supported_source_languages: HashSet<Language>,
    supported_target_languages: HashSet<Language>,
}

impl TranslationModel {
    /// Build a new `TranslationModel`
    ///
    /// # Arguments
    ///
    /// * `translation_config` - `TranslationConfig` object containing the resource references (model, vocabulary, configuration), translation options and device placement (CPU/GPU)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {     ///
    /// use rust_bert::marian::{
    ///     MarianConfigResources, MarianModelResources, MarianSourceLanguages, MarianSpmResources,
    ///     MarianTargetLanguages, MarianVocabResources,
    /// };
    /// use rust_bert::pipelines::common::ModelType;
    /// use rust_bert::pipelines::translation::{TranslationConfig, TranslationModel};
    /// use rust_bert::resources::RemoteResource;
    /// use tch::Device;
    ///
    /// let model_resource = RemoteResource::from_pretrained(MarianModelResources::ROMANCE2ENGLISH);
    /// let config_resource = RemoteResource::from_pretrained(MarianConfigResources::ROMANCE2ENGLISH);
    /// let vocab_resource = RemoteResource::from_pretrained(MarianVocabResources::ROMANCE2ENGLISH);
    /// let spm_resource = RemoteResource::from_pretrained(MarianSpmResources::ROMANCE2ENGLISH);
    ///
    /// let source_languages = MarianSourceLanguages::ROMANCE2ENGLISH;
    /// let target_languages = MarianTargetLanguages::ROMANCE2ENGLISH;
    ///
    /// let translation_config = TranslationConfig::new(
    ///     ModelType::Marian,
    ///     model_resource,
    ///     config_resource,
    ///     vocab_resource,
    ///     Some(spm_resource),
    ///     source_languages,
    ///     target_languages,
    ///     Device::cuda_if_available(),
    /// );
    /// let mut summarization_model = TranslationModel::new(translation_config)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(translation_config: TranslationConfig) -> Result<TranslationModel, RustBertError> {
        let supported_source_languages = translation_config.source_languages.clone();
        let supported_target_languages = translation_config.target_languages.clone();

        let model = TranslationOption::new(translation_config)?;

        Ok(TranslationModel {
            model,
            supported_source_languages,
            supported_target_languages,
        })
    }

    /// Build a new `TranslationModel` with a provided tokenizer.
    ///
    /// # Arguments
    ///
    /// * `translation_config` - `TranslationConfig` object containing the resource references (model, vocabulary, configuration), translation options and device placement (CPU/GPU)
    /// * `tokenizer` - `TokenizerOption` tokenizer to use for translation
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {     ///
    /// use rust_bert::marian::{
    ///     MarianConfigResources, MarianModelResources, MarianSourceLanguages, MarianSpmResources,
    ///     MarianTargetLanguages, MarianVocabResources,
    /// };
    /// use rust_bert::pipelines::common::{ModelType, TokenizerOption};
    /// use rust_bert::pipelines::translation::{TranslationConfig, TranslationModel};
    /// use rust_bert::resources::{RemoteResource, ResourceProvider};
    /// use tch::Device;
    ///
    /// let model_resource = RemoteResource::from_pretrained(MarianModelResources::ROMANCE2ENGLISH);
    /// let config_resource = RemoteResource::from_pretrained(MarianConfigResources::ROMANCE2ENGLISH);
    /// let vocab_resource = RemoteResource::from_pretrained(MarianVocabResources::ROMANCE2ENGLISH);
    /// let spm_resource = RemoteResource::from_pretrained(MarianSpmResources::ROMANCE2ENGLISH);
    ///
    /// let tokenizer = TokenizerOption::from_file(
    ///     ModelType::Marian,
    ///     vocab_resource.get_local_path()?.to_str().unwrap(),
    ///     Some(spm_resource.get_local_path()?.to_str().unwrap()),
    ///     false,
    ///     None,
    ///     None,
    /// )?;
    ///
    /// let source_languages = MarianSourceLanguages::ROMANCE2ENGLISH;
    /// let target_languages = MarianTargetLanguages::ROMANCE2ENGLISH;
    ///
    /// let translation_config = TranslationConfig::new(
    ///     ModelType::Marian,
    ///     model_resource,
    ///     config_resource,
    ///     vocab_resource,
    ///     Some(spm_resource),
    ///     source_languages,
    ///     target_languages,
    ///     Device::cuda_if_available(),
    /// );
    /// let mut summarization_model =
    ///     TranslationModel::new_with_tokenizer(translation_config, tokenizer)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new_with_tokenizer(
        translation_config: TranslationConfig,
        tokenizer: TokenizerOption,
    ) -> Result<TranslationModel, RustBertError> {
        let supported_source_languages = translation_config.source_languages.clone();
        let supported_target_languages = translation_config.target_languages.clone();

        let model = TranslationOption::new_with_tokenizer(translation_config, tokenizer)?;

        Ok(TranslationModel {
            model,
            supported_source_languages,
            supported_target_languages,
        })
    }

    /// Translates texts provided
    ///
    /// # Arguments
    /// * `input` - `&[&str]` Array of texts to summarize.
    ///
    /// # Returns
    /// * `Vec<String>` Translated texts
    ///
    /// # Example
    ///
    /// ```no_run
    /// # fn main() -> anyhow::Result<()> {
    /// use rust_bert::marian::{
    ///     MarianConfigResources, MarianModelResources, MarianSourceLanguages, MarianSpmResources,
    ///     MarianTargetLanguages, MarianVocabResources,
    /// };
    /// use rust_bert::pipelines::common::ModelType;
    /// use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
    /// use rust_bert::resources::RemoteResource;
    /// use tch::Device;
    ///
    /// let model_resource = RemoteResource::from_pretrained(MarianModelResources::ENGLISH2ROMANCE);
    /// let config_resource = RemoteResource::from_pretrained(MarianConfigResources::ENGLISH2ROMANCE);
    /// let vocab_resource = RemoteResource::from_pretrained(MarianVocabResources::ENGLISH2ROMANCE);
    /// let merges_resource = RemoteResource::from_pretrained(MarianSpmResources::ENGLISH2ROMANCE);
    /// let source_languages = MarianSourceLanguages::ENGLISH2ROMANCE;
    /// let target_languages = MarianTargetLanguages::ENGLISH2ROMANCE;
    ///
    /// let translation_config = TranslationConfig::new(
    ///     ModelType::Marian,
    ///     model_resource,
    ///     config_resource,
    ///     vocab_resource,
    ///     Some(merges_resource),
    ///     source_languages,
    ///     target_languages,
    ///     Device::cuda_if_available(),
    /// );
    /// let model = TranslationModel::new(translation_config)?;
    ///
    /// let input = ["This is a sentence to be translated"];
    /// let source_language = None;
    /// let target_language = Language::French;
    ///
    /// let output = model.translate(&input, source_language, target_language);
    /// # Ok(())
    /// # }
    /// ```
    pub fn translate<S>(
        &self,
        texts: &[S],
        source_language: impl Into<Option<Language>>,
        target_language: impl Into<Option<Language>>,
    ) -> Result<Vec<String>, RustBertError>
    where
        S: AsRef<str> + Sync,
    {
        let (prefix, forced_bos_token_id) = self.model.validate_and_get_prefix_and_forced_bos_id(
            source_language.into().as_ref(),
            target_language.into().as_ref(),
            &self.supported_source_languages,
            &self.supported_target_languages,
        )?;

        Ok(match prefix {
            Some(value) => {
                let texts = texts
                    .iter()
                    .map(|v| format!("{}{}", value, v.as_ref()))
                    .collect::<Vec<String>>();
                self.model.generate(Some(&texts), forced_bos_token_id)
            }
            None => self.model.generate(Some(texts), forced_bos_token_id),
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::marian::{
        MarianConfigResources, MarianModelResources, MarianSourceLanguages, MarianSpmResources,
        MarianTargetLanguages, MarianVocabResources,
    };
    use crate::resources::RemoteResource;

    #[test]
    #[ignore] // no need to run, compilation is enough to verify it is Send
    fn test() {
        let model_resource = RemoteResource::from_pretrained(MarianModelResources::ROMANCE2ENGLISH);
        let config_resource =
            RemoteResource::from_pretrained(MarianConfigResources::ROMANCE2ENGLISH);
        let vocab_resource = RemoteResource::from_pretrained(MarianVocabResources::ROMANCE2ENGLISH);
        let merges_resource = RemoteResource::from_pretrained(MarianSpmResources::ROMANCE2ENGLISH);

        let source_languages = MarianSourceLanguages::ROMANCE2ENGLISH;
        let target_languages = MarianTargetLanguages::ROMANCE2ENGLISH;

        let translation_config = TranslationConfig::new(
            ModelType::Marian,
            model_resource,
            config_resource,
            vocab_resource,
            Some(merges_resource),
            source_languages,
            target_languages,
            Device::cuda_if_available(),
        );
        let _: Box<dyn Send> = Box::new(TranslationModel::new(translation_config));
    }
}
