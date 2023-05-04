use tch::nn::VarStore;
use tch::{TchError, Tensor};

use crate::common::error::RustBertError;
use crate::resources::{Resource, ResourceProvider};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock, RwLockReadGuard};

#[derive(Clone)]
pub struct TensorResource {
    pub named_tensors: Arc<RwLock<HashMap<String, Tensor>>>,
}

// SAFETY: `tch::Tensor`s are neither `Send` nor `Sync`, but placing them behind an `RwLock`
// eliminates any potential for misuse. In most (all?) contexts where this would be used, there
// is no mutation of `tch::Tensor`s once they are created. The `Resource` interface to this data
// explicitly limits usage to reading by providing access through an `RwLockGuard`.
unsafe impl Send for TensorResource {}
unsafe impl Sync for TensorResource {}

impl ResourceProvider for TensorResource {
    /// Gets a wrapper referring to the in-memory weights
    ///
    /// # Returns
    ///
    /// * `Resource` referring to the named tensor data
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::sync::RwLock;
    /// use rust_bert::{
    ///     Config,
    ///     bart::{BartConfig, BartModel},
    ///     resources::{ResourceProvider, TensorResource}
    /// };
    /// use tch::{nn::VarStore};
    ///
    /// let mut var_store = VarStore::new(tch::Device::cuda_if_available());
    /// BartModel::new(
    ///     &var_store.root() / "model",
    ///     &BartConfig::from_file("/path/to/config.json"),
    /// );
    /// var_store.load_from_stream(std::io::Cursor::new(
    ///     std::fs::read("path/to/rust_model.ot").unwrap()
    /// )).unwrap();
    /// let weights_resource = TensorResource {
    ///     named_tensors: RwLock::new(var_store.variables()).into(),
    /// };
    /// let weights = weights_resource.get_resource();
    /// ```
    fn get_resource(&self) -> Result<Resource, RustBertError> {
        Ok(Resource::NamedTensors(self.named_tensors.read().unwrap()))
    }

    /// Not implemented for this resource type
    ///
    /// # Returns
    ///
    /// * `RustBertError::UnsupportedError`
    fn get_local_path(&self) -> Result<PathBuf, RustBertError> {
        Err(RustBertError::UnsupportedError)
    }
}

pub(crate) fn load_weights(
    named_tensors: RwLockReadGuard<HashMap<String, Tensor>>,
    vs: &mut VarStore,
) -> Result<(), RustBertError> {
    let mut variables = vs.variables_.lock().unwrap();
    for (name, var) in variables.named_variables.iter_mut() {
        match named_tensors.get(name) {
            Some(src) => tch::no_grad(|| var.f_copy_(src).map_err(|e| e.path_context(name)))?,
            _ => {
                return Err(RustBertError::TchError(
                    TchError::TensorNameNotFound(name.to_string(), "TensorResource".to_string())
                        .to_string(),
                ))
            }
        };
    }
    Ok(())
}

impl From<Arc<RwLock<HashMap<String, Tensor>>>> for TensorResource {
    fn from(named_tensors: Arc<RwLock<HashMap<String, Tensor>>>) -> Self {
        Self { named_tensors }
    }
}

impl From<Arc<RwLock<HashMap<String, Tensor>>>> for Box<dyn ResourceProvider> {
    fn from(named_tensors: Arc<RwLock<HashMap<String, Tensor>>>) -> Self {
        Box::new(TensorResource { named_tensors })
    }
}
