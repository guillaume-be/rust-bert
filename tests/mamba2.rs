use rust_bert::mamba2::{Mamba2Cache, Mamba2Config, Mamba2ForCausalLM};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use tch::{nn, Device, Tensor};

#[test]
fn mamba2_lm_model() -> anyhow::Result<()> {
    // Resources paths - for testing we'll use dummy paths
    let config_resource = RemoteResource::from_pretrained((
        "AntonV/mamba2-130m-hf/config",
        "https://huggingface.co/AntonV/mamba2-130m-hf/resolve/main/config.json",
    ));

    let config_path = config_resource.get_local_path()?;
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let config = Mamba2Config::from_file(config_path);

    // Initialize model
    let mamba2_model = Mamba2ForCausalLM::new(vs.root(), &config);

    // Define input - simple token ids for testing
    let input = vec![1i64, 2, 3, 4];
    let input_tensor = Tensor::from_slice(&input).unsqueeze(0).to(device);

    // Forward pass
    let model_output = mamba2_model.forward_t(Some(&input_tensor), None, None, None, false)?;

    // Check output shape
    assert_eq!(model_output.lm_logits.size(), vec![1, 4, config.vocab_size]);

    // Check that we get valid logits
    let logits = model_output.lm_logits;
    assert!(logits.abs().max().double_value(&[]) > 0.0);

    Ok(())
}

#[test]
fn mamba2_generation_with_text() -> anyhow::Result<()> {
    // Note: This test uses the exact token IDs from the Python transformers implementation
    // The model uses GPTNeoXTokenizer which produces different IDs than GPT2
    
    // Load configuration from AntonV/mamba2-130m-hf
    let config_resource = RemoteResource::from_pretrained((
        "AntonV/mamba2-130m-hf/config",
        "https://huggingface.co/AntonV/mamba2-130m-hf/resolve/main/config.json",
    ));
    let config_path = config_resource.get_local_path()?;
    let mut config = Mamba2Config::from_file(&config_path);
    
    // Use vocab_size from the actual weights (50288)
    config.vocab_size = 50288;
    
    // Use the converted pytorch_model.bin file with correct tensor names
    let weights_path = std::path::Path::new("../weights_hf_bin/AntonV/mamba2-130m-hf/pytorch_model_converted.bin").canonicalize()?;

    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    
    let mamba2_model = Mamba2ForCausalLM::new(vs.root(), &config);
    
    // Load weights
    vs.load(&weights_path)?;
    println!("Successfully loaded weights from {:?}", weights_path);

    // Use the exact token IDs from the Python implementation
    // "Hey how are you doing?" tokenized with GPTNeoXTokenizer
    let token_ids: Vec<i64> = vec![8262, 849, 403, 368, 2509, 32];
    
    println!("Input text: Hey how are you doing?");
    println!("Token IDs: {:?}", token_ids);
    
    let mut input_ids = Tensor::from_slice(&token_ids)
        .unsqueeze(0)
        .to(device);

    // Generate 10 tokens using greedy decoding (no temperature)
    let mut generated_tokens = token_ids.clone();
    let mut cache = Mamba2Cache::new(&config, config.num_hidden_layers, 1, device);
    
    // Generate 25 tokens instead of 10
    let num_tokens_to_generate = 25;
    
    println!("\nGenerating {} tokens...", num_tokens_to_generate);
    println!("Running on device: {:?}", device);
    
    for i in 0..num_tokens_to_generate {
        let output = mamba2_model.forward_t(Some(&input_ids), Some(&mut cache), None, None, false)?;
        
        // Get the last token's logits
        let next_token_logits = output.lm_logits.select(1, -1);
        
        // Greedy decoding (take argmax) - no temperature for deterministic output
        let next_token = next_token_logits.argmax(-1, false);
        let next_token_id = next_token.int64_value(&[0]);
        
        generated_tokens.push(next_token_id);
        
        // Print first 10 tokens with comparison to Python output
        if i < 10 {
            let expected = match i {
                0 => 187,
                1 => 187,
                2 => 42,
                3 => 1353,
                4 => 275,
                5 => 253,
                6 => 4766,  // Python: "middle"
                7 => 273,
                8 => 247,
                9 => 2199,  // Python: "project"
                _ => -1,
            };
            println!("Token {:2}: ID {:5} (expected: {:5})", i + 1, next_token_id, expected);
        } else {
            println!("Token {:2}: ID {:5}", i + 1, next_token_id);
        }
        
        // Append to input
        input_ids = Tensor::cat(&[input_ids, next_token.unsqueeze(1)], 1);
    }
    
    println!("\nGenerated {} new tokens", num_tokens_to_generate);
    println!("Generated token IDs: {:?}", &generated_tokens[6..]);
    
    // Verify we're on GPU if available
    if device.is_cuda() {
        println!("\n✓ Model is running on GPU");
        // Check that tensors are on GPU
        assert_eq!(input_ids.device(), device, "Input tensors should be on GPU");
    } else {
        println!("\n✓ Model is running on CPU (GPU not available)");
    }

    Ok(())
}
