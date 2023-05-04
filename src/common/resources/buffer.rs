use crate::common::error::RustBertError;
use crate::resources::{Resource, ResourceProvider};
use std::path::PathBuf;

/// # In-memory raw buffer resource
#[derive(PartialEq, Eq, Clone)]
pub struct BufferResource {
    /// The data representing the underlying resource
    pub data: Vec<u8>,
}

impl ResourceProvider for BufferResource {
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
    /// let weights_resource = BufferResource {
    ///     data: std::fs::read("path/to/rust_model.ot").unwrap(),
    /// };
    /// let weights = weights_resource.get_resource();
    /// ```
    fn get_resource(&self) -> Result<Resource, RustBertError> {
        Ok(Resource::Buffer(&self.data))
    }

    /// Not implemented for this resource type
    ///
    /// # Returns
    ///
    /// * `RustBertError::UnsupportedError`
    ///
    fn get_local_path(&self) -> Result<PathBuf, RustBertError> {
        Err(RustBertError::UnsupportedError)
    }
}

impl From<Vec<u8>> for BufferResource {
    fn from(data: Vec<u8>) -> Self {
        Self { data }
    }
}

impl From<Vec<u8>> for Box<dyn ResourceProvider> {
    fn from(data: Vec<u8>) -> Self {
        Box::new(BufferResource { data })
    }
}
