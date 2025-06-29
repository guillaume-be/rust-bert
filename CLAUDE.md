# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Building
```bash
# Standard build with automatic libtorch download
cargo build --features download-libtorch

# Build without default features
cargo build --no-default-features --features download-libtorch

# Build with ONNX support
cargo build --features download-libtorch,onnx
```

### Testing
```bash
# Run all tests
cargo test --features download-libtorch

# Run specific test modules
cargo test --package rust-bert --test bert --features download-libtorch

# Run tests with optional features
cargo test --features download-libtorch,onnx,hf-tokenizers
```

### Code Quality
```bash
# Format code
cargo fmt --all

# Check formatting without modifying
cargo fmt --all -- --check

# Run clippy linting
cargo clippy --all-targets --all-features -- -D warnings -A clippy::assign_op_pattern -A clippy::upper-case-acronyms
```

### Running Examples
```bash
# Basic example
cargo run --example sentence_embeddings --features download-libtorch

# ONNX examples require the onnx feature
cargo run --example onnx-text-generation --features download-libtorch,onnx
```

## High-Level Architecture

### Core Structure
- **src/common/**: Shared utilities including activations, embeddings, error handling, and resource management
- **src/models/**: Individual model implementations (BERT, GPT2, BART, etc.) each with their own module containing:
  - Model architecture implementation
  - Attention mechanisms
  - Embeddings
  - Encoder/decoder components
- **src/pipelines/**: High-level task-specific pipelines wrapping models for common NLP tasks:
  - Text generation, translation, summarization
  - Sequence/token classification
  - Question answering
  - Sentiment analysis
  - Named entity recognition
  - Conversation management
  - ONNX runtime support in `pipelines/onnx/`

### Key Design Patterns
1. **Resource Management**: Models use `RemoteResource` for downloading pretrained weights from Hugging Face hub, with caching in `~/.cache/.rustbert` (or `RUSTBERT_CACHE` env var)

2. **Model Loading**: All models follow a pattern of:
   - Config structs for model configuration
   - `new()` methods accepting config and variable store
   - Support for loading partial weights via `load_partial()`

3. **Pipeline Pattern**: High-level pipelines abstract model usage:
   - Builder pattern for configuration
   - Predict methods accepting batched inputs
   - Automatic tokenization and post-processing

4. **Backend Flexibility**: 
   - Primary backend: PyTorch via tch-rs bindings
   - Optional ONNX backend via ort crate for inference optimization

### Environment Setup
The project requires libtorch (PyTorch C++ API). It can be:
- Downloaded automatically with `--features download-libtorch`
- Manually installed by setting `LIBTORCH` environment variable
- For ONNX: Requires setting `ORT_DYLIB_PATH` to onnxruntime library location

### Feature Flags
- `remote`: Enable downloading models from remote sources (default)
- `download-libtorch`: Automatically download libtorch
- `onnx`: Enable ONNX runtime support
- `hf-tokenizers`: Use Hugging Face tokenizers library
- `all-tests`: Enable all integration tests