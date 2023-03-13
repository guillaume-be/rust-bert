use crate::RustBertError;
use ort::{AllocatorType, Environment, GraphOptimizationLevel, MemType, SessionBuilder};
use std::sync::Arc;

pub struct ONNXEnvironmentConfig {
    pub optimization_level: Option<GraphOptimizationLevel>,
    pub num_intra_threads: Option<i16>,
    pub num_inter_threads: Option<i16>,
    pub parallel_execution: Option<bool>,
    pub enable_memory_pattern: Option<bool>,
    pub allocator: Option<AllocatorType>,
    pub memory_type: Option<MemType>,
}

impl Default for ONNXEnvironmentConfig {
    fn default() -> Self {
        ONNXEnvironmentConfig {
            optimization_level: None,
            num_intra_threads: None,
            num_inter_threads: None,
            parallel_execution: None,
            enable_memory_pattern: None,
            allocator: None,
            memory_type: None,
        }
    }
}

impl ONNXEnvironmentConfig {
    pub(crate) fn get_session_builder(
        &self,
        environment: &Arc<Environment>,
    ) -> Result<SessionBuilder, RustBertError> {
        let mut session_builder = SessionBuilder::new(&environment)?;
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
