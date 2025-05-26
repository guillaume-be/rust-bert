# Mamba2 Model

This is the Rust implementation of the Mamba2 state space model architecture.

## Overview

Mamba2 is a state space model (SSM) that provides an efficient alternative to transformer-based architectures. This implementation is based on the paper "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" by Tri Dao and Albert Gu.

## Features

- Complete Mamba2 architecture with SSM layers
- Cache-based generation for efficient inference
- Support for models from HuggingFace (e.g., `AntonV/mamba2-130m-hf`)
- Custom JSON parsing to handle special float values (`Infinity`)
- Integration with rust-bert's generation pipeline

## Running the Example

To run the text generation example:

```bash
cargo run --example mamba2
```

**Note**: The example expects pre-trained weights to be available at `../weights_hf_bin/AntonV/mamba2-130m-hf/pytorch_model_converted.bin`. You'll need to:

1. Download the model weights from HuggingFace
2. Convert them from safetensors format to PyTorch .bin format
3. Run a conversion script to add the `backbone.` prefix and handle tied embeddings

## Model Configuration

The implementation supports loading configuration files from HuggingFace that may contain JavaScript-style special float values like `Infinity`. These are automatically handled during parsing.

## Architecture Details

The Mamba2 model consists of:
- **Mamba2Config**: Configuration structure with model hyperparameters
- **Mamba2Cache**: Cache structure for efficient autoregressive generation
- **Mamba2Mixer**: Core SSM layer with convolution and state space transformations
- **Mamba2Block**: Residual block containing normalization and mixer
- **Mamba2Model**: Stack of Mamba2 blocks with embeddings
- **Mamba2ForCausalLM**: Language modeling head on top of the base model

## Limitations

The current implementation uses a simplified SSM scan that may produce slightly different results compared to the optimized CUDA kernels used in the original implementation. The first several tokens typically match exactly, with minor divergence in later tokens due to numerical precision differences.