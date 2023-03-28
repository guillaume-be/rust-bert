use crate::{
    m2m_100::M2M100Generator,
    pipelines::translation::Language::{self, *},
};

pub struct NLLBResources;
pub struct NLLBConfigResources;
pub struct NLLBVocabResources;
pub struct NLLBMergeResources;
pub struct NLLBLanguages;
pub struct NLLBSpecialMap;

impl NLLBResources {
    pub const NLLB_600M_DISTILLED: (&'static str, &'static str) = (
        "nllb200-distilled-600m/model",
        "https://huggingface.co/datasets/vpermilp/nllb-200-distilled-600M-rust/resolve/main/rust_model.ot",
    );

    pub const NLLB_1_3B: (&'static str, &'static str) = (
        "nllb200-1_3b/model",
        "https://huggingface.co/datasets/vpermilp/nllb-200-1.3B-rust/resolve/main/rust_model.ot",
    );
}

impl NLLBConfigResources {
    pub const NLLB_600M_DISTILLED: (&'static str, &'static str) = (
        "nllb200-distilled-600m/config",
        "https://huggingface.co/datasets/vpermilp/nllb-200-distilled-600M-rust/raw/main/config.json",
    );

    pub const NLLB_1_3B: (&'static str, &'static str) = (
        "nllb200-1_3b/config",
        "https://huggingface.co/datasets/vpermilp/nllb-200-1.3B-rust/raw/main/config.json",
    );
}

impl NLLBVocabResources {
    pub const NLLB_600M_DISTILLED: (&'static str, &'static str) = (
        "nllb200-distilled-600m/vocab",
        "https://huggingface.co/datasets/vpermilp/nllb-200-distilled-600M-rust/resolve/main/tokenizer.json",
    );

    pub const NLLB_1_3B: (&'static str, &'static str) = (
        "nllb200-1_3b/vocab",
        "https://huggingface.co/datasets/vpermilp/nllb-200-1.3B-rust/resolve/main/tokenizer.json",
    );
}

impl NLLBMergeResources {
    pub const NLLB_600M_DISTILLED: (&'static str, &'static str) = (
        "nllb200-distilled-600m/merge",
        "https://huggingface.co/datasets/vpermilp/nllb-200-distilled-600M-rust/resolve/main/sentencepiece.bpe.model",
    );

    pub const NLLB_1_3B: (&'static str, &'static str) = (
        "nllb200-1_3b/merge",
        "https://huggingface.co/datasets/vpermilp/nllb-200-1.3B-rust/resolve/main/sentencepiece.bpe.model",
    );
}

impl NLLBSpecialMap {
    pub const NLLB_600M_DISTILLED: (&'static str, &'static str) = (
        "nllb200-distilled-600m/special",
        "htps://huggingface.co/datasets/vpermilp/nllb-200-distilled-600M-rust/raw/main/special_tokens_map.json",
    );

    pub const NLLB_1_3B: (&'static str, &'static str) = (
        "nllb200-1_3b/special",
        "https://huggingface.co/datasets/vpermilp/nllb-200-1.3B-rust/raw/main/special_tokens_map.json",
    );
}

impl NLLBLanguages {
    #[rustfmt::skip]
    pub const NLLB: [Language; 201] = [
        Afrikaans, Danish, Dutch, German, English, Icelandic, Luxembourgish, Swedish,
        Asturian, Catalan, French, Galician, Italian, Occitan, Portuguese, Romanian, Spanish,
        Belarusian, Bosnian, Bulgarian, Croatian, Czech, Macedonian, Polish, Russian, Serbian, Slovak,
        Slovenian, Ukrainian, Estonian, Finnish, Hungarian, Latvian, Lithuanian, Albanian,
        Armenian, Georgian, Greek, Irish, ScottishGaelic, Welsh, Bashkir, Kazakh,
        Turkish, Uzbek, Japanese, Korean, Vietnamese, Bengali, Gujarati, Hindi, Kannada,
        Marathi, Oriya, Panjabi, Sindhi, Sinhala, Urdu, Tamil, Cebuano, Iloko, Indonesian,
        Javanese, Malay, Malayalam, Sundanese, Tagalog, Burmese, CentralKhmer, Lao, Thai, Hebrew, Amharic,
        Hausa, Igbo, Lingala, Luganda, NorthernSotho, Somali, Swahili, Swati, Tswana, Wolof, Xhosa,
        Yoruba, Zulu, HaitianCreole, Achinese, MesopotamianArabic, TaizziAdeniArabic, TunisianArabic, SouthLevantineArabic, Akan, NorthLevantineArabic, Arabic,
        NajdiArabic, MoroccanArabic, EgyptianArabic, Assamese, Awadhi, CentralAymara, SouthAzerbaijani, NorthAzerbaijani, Bambara, Balinese,
        Bemba, Bhojpuri, Banjar, Tibetan, Buginese, Chokwe, CentralKurdish, CrimeanTatar, SouthwesternDinka, Dyula, Dzongkha,
        Esperanto, Basque, Ewe, Faroese, Fijian, Fon, Friulian, NigerianFulfulde, WestCentralOromo, Guarani, Haitian,
        Chhattisgarhi, Kabyle, Kachin, Kamba, Kashmiri, Kabiye, Kabuverdianu, HalhMongolian, Khmer, Kikuyu,
        Kinyarwanda, Kirghiz, Kimbundu, NorthernKurdish, CentralKanuri, Kongo, Ligurian, Limburgan, Lombard, Latgalian,
        LubaLulua, Ganda, Luo, Lushai, Magahi, Maithili, Minangkabau, Maltese, Manipuri, Mossi,
        Maori, NorwegianNynorsk, NorwegianBokmal, Pedi, Nuer, Nyanja, Odia, Pangasinan, Papiamento, SouthernPashto, IranianPersian,
        PlateauMalagasy, Dari, AyacuchoQuechua, Rundi, Sango, Sanskrit, Santali, Sicilian, Shan, Samoan,
        Shona, SouthernSotho, Sardinian, Silesian, Tamasheq, Tatar, Telugu, Tajik, Tigrinya, TokPisin,
        Tsonga, Turkmen, Tumbuka, Twi, CentralAtlasTamazight, Uighur, Umbundu, NorthernUzbek, Venetian, Waray,
        EasternYiddish, YueChinese, Chinese,
    ];
}

pub type NLLBGenerator = M2M100Generator;
