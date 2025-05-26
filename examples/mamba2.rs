use anyhow::Result;
use rust_bert::mamba2::{Mamba2Cache, Mamba2Config, Mamba2ForCausalLM};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{GptNeoXTokenizer, Tokenizer};
use tch::{nn, Device, Kind, Tensor};

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
    
    // Load tokenizer from the HuggingFace tokenizer.json file
    let tokenizer_resource = RemoteResource::from_pretrained((
        "AntonV/mamba2-130m-hf/tokenizer.json",
        "https://huggingface.co/AntonV/mamba2-130m-hf/resolve/main/tokenizer.json",
    ));
    
    // Initialize the tokenizer using the new from_tokenizer_json method
    let tokenizer = GptNeoXTokenizer::from_tokenizer_json(
        tokenizer_resource.get_local_path()?,
        false,  // lower_case
        false,  // add_prefix_space
        false,  // add_bos_token
        false,  // add_eos_token
    )?;
    
    // Input text
    let input_text = "Hey how are you doing?";
    println!("Input: {}", input_text);
    
    // Tokenize the input
    let encoding = tokenizer.encode(
        input_text,
        None,
        128,
        &rust_tokenizers::tokenizer::TruncationStrategy::LongestFirst,
        0,
    );
    
    println!("Input tokens: {:?}", encoding.token_ids);
    
    let input_ids = Tensor::from_slice(&encoding.token_ids)
        .unsqueeze(0)
        .to(device);
    
    // Generate text until EOS token or max length
    let mut cache = Mamba2Cache::new(&config, config.num_hidden_layers, 1, device);
    let mut generated = input_ids.shallow_clone();
    let mut generated_tokens = encoding.token_ids.clone();
    let eos_token_id = config.eos_token_id.unwrap_or(0);
    let max_length = 100; // Medium length to balance speed and EOS generation
    let temperature = 0.8; // Temperature for sampling
    let mut consecutive_punctuation = 0;
    
    println!("Generating text...\n");
    
    while generated.size()[1] < max_length {
        let output = model.forward_t(Some(&generated), Some(&mut cache), None, None, false)?;
        let logits = output.lm_logits.select(1, -1);
        
        // Apply temperature
        let scaled_logits = &logits / temperature;
        let probs = scaled_logits.softmax(-1, Kind::Float);
        
        // Sample from the distribution
        let next_token = probs.multinomial(1, true).squeeze_dim(1);
        let next_token_id = next_token.int64_value(&[0]);
        
        if next_token_id == eos_token_id {
            println!("\nEOS token encountered at position {}", generated_tokens.len());
            break;
        }
        
        generated_tokens.push(next_token_id);
        generated = Tensor::cat(&[generated, next_token.unsqueeze(1)], 1);
        
        // Decode the last token to check for stopping conditions
        let last_token_text = tokenizer.decode(&[next_token_id], false, false);
        
        // Check for sentence-ending punctuation
        if last_token_text.contains('.') || last_token_text.contains('!') || last_token_text.contains('?') {
            consecutive_punctuation += 1;
            if consecutive_punctuation >= 2 || generated_tokens.len() > 30 {
                println!("\nStopping at natural sentence boundary.");
                break;
            }
        } else {
            consecutive_punctuation = 0;
        }
        
        // Print progress with the actual text
        if generated_tokens.len() % 10 == 0 {
            let partial_text = tokenizer.decode(&generated_tokens, false, false);
            println!("\nPartial text ({} tokens): {}", generated_tokens.len(), partial_text);
        }
    }
    
    // Decode the generated tokens
    let generated_text = tokenizer.decode(
        &generated_tokens,
        false,  // skip_special_tokens
        false,  // clean_up_tokenization_spaces
    );
    
    // Display results
    println!("Generated {} new tokens", generated_tokens.len() - encoding.token_ids.len());
    println!("\nGenerated text:");
    println!("\"{}\"", generated_text);
    
    Ok(())
}