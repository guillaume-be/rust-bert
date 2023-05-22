use crate::common::error::RustBertError;
use crate::resources::{Resource, ResourceProvider};
use std::path::PathBuf;

/// # In-memory raw buffer resource
#[derive(PartialEq, Eq, Clone)]
pub struct BufferResource {
    /// The data representing the underlying resource
    data: Vec<u8>,
    is_valid: bool,
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
        if !self.is_valid {
            Ok(Resource::Buffer(&self.data))
        } else {
            Err(RustBertError::ValueError(
                "Resource has been consumed and its internal buffer is invalid".to_string(),
            ))
        }
    }

    /// Mark if a resource has been consumed.
    ///
    /// For some `ResourceProvider`, the buffer is consumed when loading the weights,
    /// meaning they cannot be loaded twice (a new resource needs to be created)
    fn mark_consumed(&mut self) {
        self.is_valid = false;
    }

    /// Check if a resource is still valid for loading.
    ///
    /// For some `ResourceProvider`, the buffer is consumed when loading the weights,
    /// meaning they cannot be loaded twice (a new resource needs to be created)
    fn is_valid(&self) -> bool {
        self.is_valid
    }
}

impl From<Vec<u8>> for BufferResource {
    fn from(data: Vec<u8>) -> Self {
        Self {
            data,
            is_valid: false,
        }
    }
}

impl From<Vec<u8>> for Box<dyn ResourceProvider> {
    fn from(data: Vec<u8>) -> Self {
        Box::new(BufferResource {
            data,
            is_valid: false,
        })
    }
}
