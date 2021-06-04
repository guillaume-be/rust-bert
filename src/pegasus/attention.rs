// Copyright 2021, Google and The HuggingFace Inc. team. All rights reserved.
// Copyright 2021 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::bart::LayerState as BartLayerState;

/// # Cache for Pegasus attention layers
/// Stores the cached value of key, value and key padding mask to avoid recalculation (e.g. at each generation step)
/// Identical to BART cache (type alias).
pub type LayerState = BartLayerState;
