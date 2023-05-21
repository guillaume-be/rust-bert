import argparse
import numpy as np
import subprocess
import sys
import torch

from pathlib import Path
from torch import Tensor

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
