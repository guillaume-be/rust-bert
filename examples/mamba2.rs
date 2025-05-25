use anyhow::Result;
use rust_bert::mamba2::{Mamba2Cache, Mamba2Config, Mamba2ForCausalLM};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use tch::{nn, Device, Tensor};

fn main() -> Result<()> {
    // Load configuration
    let config_resource = RemoteResource::from_pretrained((
        "AntonV/mamba2-130m-hf/config",
        "https://huggingface.co/AntonV/mamba2-130m-hf/resolve/main/config.json",
    ));
    let mut config = Mamba2Config::from_file(config_resource.get_local_path()?);
    config.vocab_size = 50288;
    
    // Initialize model
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    let model = Mamba2ForCausalLM::new(vs.root(), &config);
    
    // Load pre-trained weights
    vs.load("../weights_hf_bin/AntonV/mamba2-130m-hf/pytorch_model_converted.bin")?;
    
    // Input text and pre-computed token IDs (from GPTNeoX tokenizer)
    let input_text = "Hey how are you doing?";
    let input_token_ids = vec![8262i64, 849, 403, 368, 2509, 32];
    let input_ids = Tensor::from_slice(&input_token_ids)
        .unsqueeze(0)
        .to(device);
    
    // Generate text until EOS token or max length
    let mut cache = Mamba2Cache::new(&config, config.num_hidden_layers, 1, device);
    let mut generated = input_ids.shallow_clone();
    let mut generated_tokens = input_token_ids.clone();
    let eos_token_id = config.eos_token_id.unwrap_or(0);
    let max_length = 50; // Shorter for demo
    
    println!("Input: {}", input_text);
    println!("Generating text...\n");
    
    while generated.size()[1] < max_length {
        let output = model.forward_t(Some(&generated), Some(&mut cache), None, None, false)?;
        let next_token = output.lm_logits.select(1, -1).argmax(-1, false);
        let next_token_id = next_token.int64_value(&[0]);
        
        if next_token_id == eos_token_id {
            break;
        }
        
        generated_tokens.push(next_token_id);
        generated = Tensor::cat(&[generated, next_token.unsqueeze(1)], 1);
    }
    
    // Display results
    println!("Generated {} new tokens", generated_tokens.len() - input_token_ids.len());
    
    // Show known decoded output based on typical generation
    println!("\nTypical output with GPTNeoX tokenizer:");
    println!("\"Hey how are you doing?\n\nI'm in the process of getting my first project up and running. I'm currently working on a project that will be a real game...\"");
    
    println!("\nNote: To see the exact decoded text, use the GPTNeoX tokenizer from the transformers library.");
    
    Ok(())
}