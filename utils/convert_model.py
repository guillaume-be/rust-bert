from pathlib import Path
import numpy as np
import torch
import subprocess
import argparse
import sys

from torch import Tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", help="Absolute path to the Pytorch weights file to convert")
    parser.add_argument("--skip_embeddings", action="store_true", help="Skip shared embeddings / language model head")
    args = parser.parse_args()

    source_file = Path(args.source_file)
    target_folder = source_file.parent

    weights = torch.load(str(source_file), map_location='cpu')

    nps = {}
    for k, v in weights.items():
        k = k.replace("gamma", "weight").replace("beta", "bias")
        if args.skip_embeddings:
            if k in {"lm_head.weight", "model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight"}:
                continue
        if isinstance(v, Tensor):
            nps[k] = np.ascontiguousarray(v.cpu().numpy().astype(np.float32))
            print(f'converted {k} - {str(sys.getsizeof(nps[k]))} bytes')
        else:
            print(f'skipped non-tensor object: {k}')
    np.savez(target_folder / 'model.npz', **nps)

    source = str(target_folder / 'model.npz')
    target = str(target_folder / 'rust_model.ot')

    toml_location = (Path(__file__).resolve() / '..' / '..' / 'Cargo.toml').resolve()
    subprocess.run(
        ['cargo', 'run', '--bin=convert-tensor', '--manifest-path=%s' % toml_location, '--', source, target],
    )
