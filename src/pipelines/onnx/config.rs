use crate::RustBertError;
use ort::{
    AllocatorType, Environment, ExecutionProvider, GraphOptimizationLevel, MemType, SessionBuilder,
};
use std::sync::Arc;

pub static INPUT_IDS_NAME: &str = "input_ids";
pub static ATTENTION_MASK_NAME: &str = "attention_mask";
pub static ENCODER_HIDDEN_STATES_NAME: &str = "encoder_hidden_states";
pub static ENCODER_ATTENTION_MASK_NAME: &str = "encoder_attention_mask";
pub static TOKEN_TYPE_IDS: &str = "token_type_ids";
pub static POSITION_IDS: &str = "position_ids";
pub static INPUT_EMBEDS: &str = "input_embeds";
pub static LAST_HIDDEN_STATE: &str = "last_hidden_state";

#[derive(Default)]
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
    pub(crate) fn get_session_builder(
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
            session_builder = session_builder.with_allocator(allocator.clone())?;
        }
        if let Some(memory_type) = &self.memory_type {
            session_builder = session_builder.with_memory_type(memory_type.clone())?;
        }
        Ok(session_builder)
    }
}
