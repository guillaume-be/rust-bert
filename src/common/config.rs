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

use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// # Utility to deserialize JSON config files
pub trait Config
where
    for<'de> Self: Deserialize<'de>,
{
    /// Loads a `Config` object from a JSON file. The format is expected to be aligned with the [Transformers library](https://github.com/huggingface/transformers) configuration files for each model.
    /// The parsing will fail if non-optional keys expected by the model are missing.
    ///
    /// # Arguments
    ///
    /// * `path` - `Path` to the configuration JSON file.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::gpt2::Gpt2Config;
    /// use rust_bert::Config;
    /// use std::path::Path;
    ///
    /// let config_path = Path::new("path/to/config.json");
    /// let config = Gpt2Config::from_file(config_path);
    /// ```
    fn from_file<P: AsRef<Path>>(path: P) -> Self {
        let f = File::open(path).expect("Could not open configuration file.");
        let br = BufReader::new(f);
        let config: Self = serde_json::from_reader(br).expect("could not parse configuration");
        config
    }
}
