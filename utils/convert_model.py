# Copyright 2019-2023 Guillaume Becquin
# Copyright 2023 https://github.com/starkat99/half-rs
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import numpy as np
import subprocess
import sys
import torch

from pathlib import Path
from torch import Tensor


def get_bf16_repr(input_tensor: torch.Tensor) -> np.ndarray:
    """Convert a bfloat16 tensor to an equivalent byte representation in Numpy.
    This is a vectorized implementation inspired from https://github.com/starkat99/half-rs/blob/main/src/bfloat/convert.rs
    (shared under Apache 2.0 license at https://github.com/starkat99/half-rs/blob/main/LICENSES/Apache-2.0.txt)
    """
    v_fp32 = input_tensor.cpu().float().numpy()
    byte_array = np.frombuffer(v_fp32.tobytes(), dtype=np.uint32)
    nan_value = np.logical_or(np.right_shift(byte_array, 16), 0x0040)
    nan_mask = np.logical_and(byte_array, 0x7FFF_FFFF) > 0x7F80_0000
    round_bit = 0x0000_8000
    output_val = np.right_shift(byte_array, 16)
    threshold_mask = (np.logical_and(byte_array, round_bit) != 0) & (np.logical_and(byte_array, (3*round_bit-1)) != 0)
    output = np.where(nan_mask, nan_value, np.where(threshold_mask, output_val+1, output_val)).astype(np.uint16)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_file", nargs="+", help="Absolute path to the Pytorch weights file to convert"
    )
    parser.add_argument(
        "--skip_embeddings",
        action="store_true",
        help="Skip shared embeddings",
    )
    parser.add_argument(
        "--skip_lm_head", action="store_true", help="Skip language model head"
    )
    parser.add_argument("--prefix", help="Add a prefix on weight names")
    parser.add_argument(
        "--suffix",
        action="store_true",
        help="Split weight names on '.' and keep only last part",
    )
    parser.add_argument(
        "--dtype",
        help="Convert weights to a specific numpy DataType (float32, float16, ...)",
    )
    parser.add_argument(
        "--download_libtorch",
        action="store_true",
        help="Use this flag to enable automatic download of the libtorch library.",
    )
    args = parser.parse_args()

    nps = {}
    target_folder = Path(args.source_file[0]).parent

    for source_file in args.source_file:
        source_file = Path(source_file)
        weights = torch.load(str(source_file), map_location="cpu")
        
        for k, v in weights.items():
            k = k.replace("gamma", "weight").replace("beta", "bias")
            if args.skip_embeddings:
                if k in {
                    "model.encoder.embed_tokens.weight",
                    "encoder.embed_tokens.weight",
                    "model.decoder.embed_tokens.weight",
                    "decoder.embed_tokens.weight",
                }:
                    continue
            if args.skip_lm_head:
                if k in {
                    "lm_head.weight",
                }:
                    continue
            if args.prefix:
                k = args.prefix + k
            if args.suffix:
                k = k.split(".")[-1]
            if isinstance(v, Tensor):
                if v.dtype == torch.bfloat16:
                    tensor = get_bf16_repr(v)
                else:
                    tensor = v.cpu().numpy()
                if args.dtype is not None:
                    nps[k] = np.ascontiguousarray(tensor.astype(np.dtype(args.dtype)))
                else:
                    nps[k] = np.ascontiguousarray(tensor)
                print(f"converted {k} - {str(sys.getsizeof(nps[k]))} bytes")
            else:
                print(f"skipped non-tensor object: {k}")
    np.savez(target_folder / "model.npz", **nps)

    source = str(target_folder / "model.npz")
    target = str(target_folder / "rust_model.ot")

    toml_location = (Path(__file__).resolve() / ".." / ".." / "Cargo.toml").resolve()
    cargo_args = [
        "cargo",
        "run",
        "--bin=convert-tensor",
        "--manifest-path=%s" % toml_location,
        "--",
        source,
        target,
        ]
    if args.download_libtorch:
        cargo_args += ["--features", "download-libtorch"]
    subprocess.run(cargo_args)
