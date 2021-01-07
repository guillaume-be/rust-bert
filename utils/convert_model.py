from pathlib import Path
import numpy as np
import torch
import subprocess


source_file = Path("path/to/pytorch_model.bin")
target_folder = source_file.parent

weights = torch.load(str(source_file), map_location='cpu')

nps = {}
for k, v in weights.items():
    k = k.replace("gamma", "weight").replace("beta", "bias")
    nps[k] = np.ascontiguousarray(v.cpu().numpy())

np.savez(target_folder / 'model.npz', **nps)

source = str(target_folder / 'model.npz')
target = str(target_folder / 'rust_model.ot')

toml_location = (Path(__file__).resolve() / '..' / '..' / 'Cargo.toml').resolve()

subprocess.call(
    ['cargo', 'run', '--bin=convert-tensor', '--manifest-path=%s' % toml_location, '--', source, target])
