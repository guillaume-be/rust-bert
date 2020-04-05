from transformers import BART_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.configuration_bart import BART_PRETRAINED_CONFIG_ARCHIVE_MAP
from transformers.tokenization_bart import vocab_url, merges_url
from transformers.file_utils import get_from_cache
from pathlib import Path
import shutil
import os
import numpy as np
import torch
import subprocess

config_path = BART_PRETRAINED_CONFIG_ARCHIVE_MAP['bart-large']
vocab_path = vocab_url
merges_path = merges_url
weights_path = BART_PRETRAINED_MODEL_ARCHIVE_MAP['bart-large']

target_path = Path.home() / 'rustbert' / 'bart-large'

temp_config = get_from_cache(config_path)
temp_vocab = get_from_cache(vocab_path)
temp_merges = get_from_cache(merges_path)
temp_weights = get_from_cache(weights_path)

os.makedirs(str(target_path), exist_ok=True)

config_path = str(target_path / 'config.json')
vocab_path = str(target_path / 'vocab.txt')
merges_path = str(target_path / 'merges.txt')
model_path = str(target_path / 'model.bin')

shutil.copy(temp_config, config_path)
shutil.copy(temp_vocab, vocab_path)
shutil.copy(temp_merges, merges_path)
shutil.copy(temp_weights, model_path)

weights = torch.load(temp_weights, map_location='cpu')
nps = {}
for k, v in weights.items():
    k = k.replace("gamma", "weight").replace("beta", "bias")
    if '.shared' in k:
        nps[k.replace('.shared', '.shared_encoder')] = np.ascontiguousarray(v.cpu().numpy())
        nps[k.replace('.shared', '.shared_decoder')] = np.ascontiguousarray(v.cpu().numpy())
    else:
        nps[k] = np.ascontiguousarray(v.cpu().numpy())

np.savez(target_path / 'model.npz', **nps)

source = str(target_path / 'model.npz')
target = str(target_path / 'model.ot')

toml_location = (Path(__file__).resolve() / '..' / '..' / 'Cargo.toml').resolve()

subprocess.call(
    ['cargo', 'run', '--bin=convert-tensor', '--manifest-path=%s' % toml_location, '--', source, target])
