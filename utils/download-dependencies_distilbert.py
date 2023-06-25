import os
import subprocess
from pathlib import Path

import numpy as np
import requests
import torch

if __name__ == "__main__":
    target_path = Path.home() / "rustbert" / "distilbert"
    os.makedirs(str(target_path), exist_ok=True)

    weights_url = "https://huggingface.co/sshleifer/tiny-distilbert-base-cased/resolve/main/pytorch_model.bin"
    r = requests.get(weights_url, allow_redirects=True)
    (target_path / "pytorch_model.bin").open("wb").write(r.content)

    weights = torch.load(target_path / "pytorch_model.bin", map_location="cpu")
    nps = {}
    for k, v in weights.items():
        nps[k] = np.ascontiguousarray(v.cpu().numpy())

    np.savez(target_path / "model.npz", **nps)

    source = str(target_path / "model.npz")
    target = str(target_path / "model.ot")

    toml_location = (Path(__file__).resolve() / ".." / ".." / "Cargo.toml").resolve()

    subprocess.call(
        [
            "cargo",
            "run",
            "--bin=convert-tensor",
            "--features",
            "download-libtorch",
            "--manifest-path=%s" % toml_location,
            "--",
            source,
            target,
        ]
    )

    os.remove(str(target_path / "pytorch_model.bin"))
    os.remove(str(target_path / "model.npz"))

    assert (target_path / "model.ot").exists(), "Conversion of the model failed."
