from transformers.file_utils import get_from_cache, S3_BUCKET_PREFIX
from transformers.pipelines import SUPPORTED_TASKS
from pathlib import Path
import shutil
import os
import numpy as np
import torch
import subprocess

ROOT_PATH = S3_BUCKET_PREFIX + '/' + SUPPORTED_TASKS['ner']['default']['model']['pt']

config_path = ROOT_PATH + '/config.json'
vocab_path = ROOT_PATH + '/vocab.txt'
weights_path = ROOT_PATH + '/pytorch_model.bin'

target_path = Path.home() / 'rustbert' / 'bert-ner'

temp_config = get_from_cache(config_path)
temp_vocab = get_from_cache(vocab_path)
temp_weights = get_from_cache(weights_path)

os.makedirs(str(target_path), exist_ok=True)

config_path = str(target_path / 'config.json')
vocab_path = str(target_path / 'vocab.txt')
model_path = str(target_path / 'model.bin')

shutil.copy(temp_config, config_path)
shutil.copy(temp_vocab, vocab_path)
shutil.copy(temp_weights, model_path)

weights = torch.load(temp_weights, map_location='cpu')
nps = {}
for k, v in weights.items():
    k = k.replace("gamma", "weight").replace("beta", "bias")
    nps[k] = np.ascontiguousarray(v.cpu().numpy())

np.savez(target_path / 'model.npz', **nps)

source = str(target_path / 'model.npz')
target = str(target_path / 'model.ot')

toml_location = (Path(__file__).resolve() / '..' / '..' / 'Cargo.toml').resolve()

subprocess.call(
    ['cargo', 'run', '--bin=convert-tensor', '--manifest-path=%s' % toml_location, '--', source, target])

os.remove(str(target_path / 'model.bin'))
os.remove(str(target_path / 'model.npz'))