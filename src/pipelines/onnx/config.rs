/// # Configuration for ONNX environment and sessions
use crate::RustBertError;
use ort::execution_providers::{CPUExecutionProviderOptions, CUDAExecutionProviderOptions};
use ort::{
    AllocatorType, Environment, ExecutionProvider, GraphOptimizationLevel, MemType, SessionBuilder,
};
use std::sync::Arc;
use tch::Device;

pub(crate) static INPUT_IDS_NAME: &str = "input_ids";
pub(crate) static ATTENTION_MASK_NAME: &str = "attention_mask";
pub(crate) static ENCODER_HIDDEN_STATES_NAME: &str = "encoder_hidden_states";
pub(crate) static ENCODER_ATTENTION_MASK_NAME: &str = "encoder_attention_mask";
pub(crate) static TOKEN_TYPE_IDS: &str = "token_type_ids";
pub(crate) static POSITION_IDS: &str = "position_ids";
pub(crate) static INPUT_EMBEDS: &str = "input_embeds";
pub(crate) static LAST_HIDDEN_STATE: &str = "last_hidden_state";
pub(crate) static LOGITS: &str = "logits";
pub(crate) static START_LOGITS: &str = "start_logits";
pub(crate) static END_LOGITS: &str = "end_logits";

#[derive(Default)]
/// # ONNX Environment configuration
/// See <https://onnxruntime.ai/docs/api/python/api_summary.html#sessionoptions>
pub struct ONNXEnvironmentConfig {
    pub optimization_level: Option<GraphOptimizationLevel>,
    pub execution_providers: Option<Vec<ExecutionProvider>>,
    pub num_intra_threads: Option<i16>,
    pub num_inter_threads: Option<i16>,
    pub parallel_execution: Option<bool>,
    pub enable_memory_pattern: Option<bool>,
    pub allocator: Option<AllocatorType>,
    pub memory_type: Option<MemType>,
}

impl ONNXEnvironmentConfig {
    /// Create a new `ONNXEnvironmentConfig` from a `tch::Device`.
    /// This helper function maps torch device to ONNXRuntime execution providers
    pub fn from_device(device: Device) -> Self {
        let mut execution_providers = Vec::new();
        if let Device::Cuda(device) = device {
            execution_providers.push(ExecutionProvider::CUDA(CUDAExecutionProviderOptions {
                device_id: device as u32,
                ..Default::default()
            }));
        };
        execution_providers.push(ExecutionProvider::CPU(
            CPUExecutionProviderOptions::default(),
        ));
        ONNXEnvironmentConfig {
            execution_providers: Some(execution_providers),
            ..Default::default()
        }
    }

    ///Build a session builder from an `ONNXEnvironmentConfig`.
    pub fn get_session_builder(
        &self,
        environment: &Arc<Environment>,
    ) -> Result<SessionBuilder, RustBertError> {
        let mut session_builder = SessionBuilder::new(environment)?;
        match &self.optimization_level {
            Some(GraphOptimizationLevel::Level3) | None => {}
            Some(GraphOptimizationLevel::Level2) => {
                session_builder =
                    session_builder.with_optimization_level(GraphOptimizationLevel::Level2)?
            }
            Some(GraphOptimizationLevel::Level1) => {
                session_builder =
                    session_builder.with_optimization_level(GraphOptimizationLevel::Level1)?
            }
            Some(GraphOptimizationLevel::Disable) => {
                session_builder =
                    session_builder.with_optimization_level(GraphOptimizationLevel::Disable)?
            }
        }
        if let Some(num_intra_threads) = self.num_intra_threads {
            session_builder = session_builder.with_intra_threads(num_intra_threads)?;
        }
        if let Some(num_inter_threads) = self.num_inter_threads {
            session_builder = session_builder.with_inter_threads(num_inter_threads)?;
        }
        if let Some(parallel_execution) = self.parallel_execution {
            session_builder = session_builder.with_parallel_execution(parallel_execution)?;
        }
        if let Some(enable_memory_pattern) = self.enable_memory_pattern {
            session_builder = session_builder.with_memory_pattern(enable_memory_pattern)?;
        }
        if let Some(allocator) = &self.allocator {
            session_builder = session_builder.with_allocator(*allocator)?;
        }
        if let Some(memory_type) = &self.memory_type {
            session_builder = session_builder.with_memory_type(*memory_type)?;
        }
        Ok(session_builder)
    }

    ///Build an ONNXEnvironment from an `ONNXEnvironmentConfig`.
    pub fn get_environment(&self) -> Result<Arc<Environment>, RustBertError> {
        Ok(Arc::new(
            Environment::builder()
                .with_name("Default environment")
                .with_execution_providers(self.execution_providers.clone().unwrap_or(vec![
                    ExecutionProvider::CPU(CPUExecutionProviderOptions::default()),
                ]))
                .build()?,
        ))
    }
}
