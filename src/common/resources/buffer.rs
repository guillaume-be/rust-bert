use crate::common::error::RustBertError;
use crate::resources::{Resource, ResourceProvider};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// # In-memory raw buffer resource
#[derive(Debug)]
pub struct BufferResource {
    /// The data representing the underlying resource
    pub data: Arc<RwLock<Vec<u8>>>,
}

impl ResourceProvider for BufferResource {
    /// Not implemented for this resource type
    ///
    /// # Returns
    ///
    /// * `RustBertError::UnsupportedError`
    fn get_local_path(&self) -> Result<PathBuf, RustBertError> {
        Err(RustBertError::UnsupportedError)
    }

    /// Gets a wrapper referring to the in-memory resource.
    ///
    /// # Returns
    ///
    /// * `Resource` referring to the resource data
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::resources::{BufferResource, ResourceProvider};
    /// let data = std::fs::read("path/to/rust_model.ot").unwrap();
    /// let weights_resource = BufferResource::from(data);
    /// let weights = weights_resource.get_resource();
    /// ```
    fn get_resource(&self) -> Result<Resource, RustBertError> {
        Ok(Resource::Buffer(self.data.write().unwrap()))
    }
}

impl From<Vec<u8>> for BufferResource {
    fn from(data: Vec<u8>) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
        }
    }
}

impl From<Vec<u8>> for Box<dyn ResourceProvider> {
    fn from(data: Vec<u8>) -> Self {
        Box::new(BufferResource {
            data: Arc::new(RwLock::new(data)),
        })
    }
}

impl From<RwLock<Vec<u8>>> for BufferResource {
    fn from(lock: RwLock<Vec<u8>>) -> Self {
        Self {
            data: Arc::new(lock),
        }
    }
}

impl From<RwLock<Vec<u8>>> for Box<dyn ResourceProvider> {
    fn from(lock: RwLock<Vec<u8>>) -> Self {
        Box::new(BufferResource {
            data: Arc::new(lock),
        })
    }
}
