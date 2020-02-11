from transformers import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.tokenization_distilbert import PRETRAINED_VOCAB_FILES_MAP
from transformers.file_utils import get_from_cache
from pathlib import Path
import shutil
import os
import numpy as np
import torch
import subprocess

config_path = DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP["distilbert-base-uncased-finetuned-sst-2-english"]
vocab_path = PRETRAINED_VOCAB_FILES_MAP["vocab_file"]["distilbert-base-uncased"]
weights_path = DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP["distilbert-base-uncased-finetuned-sst-2-english"]

target_path = Path.home() / 'rustbert'

temp_config = get_from_cache(config_path)
temp_vocab = get_from_cache(vocab_path)
temp_weights = get_from_cache(weights_path)

os.makedirs(target_path, exist_ok=True)
shutil.copy(temp_config, target_path / 'config.json')
shutil.copy(temp_vocab, target_path / 'vocab.txt')
shutil.copy(temp_weights, target_path / 'model.bin')

weights = torch.load(temp_weights)
nps = {}
for k, v in weights.items():
    nps[k] = v.cpu().numpy()

np.savez(target_path / 'model.npz', **nps)

source = str(target_path / 'model.npz')
target = str(target_path / 'model.ot')

toml_location = (Path(__file__).resolve() / '..' / '..' / 'Cargo.toml').resolve()

subprocess.call(
    ['cargo', '+nightly', 'run', '--bin=convert-tensor', f'--manifest-path={toml_location}', '--', source,
     target])
