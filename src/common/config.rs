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


use std::path::Path;
use std::fs::File;
use std::io::BufReader;
use serde::Deserialize;

pub trait Config<T>
    where for<'de> T: Deserialize<'de> {
    fn from_file(path: &Path) -> T {
        let f = File::open(path).expect("Could not open configuration file.");
        let br = BufReader::new(f);
        let config: T = serde_json::from_reader(br).expect("could not parse configuration");
        config
    }
}