use crate::common::error::RustBertError;
use crate::resources::{Resource, ResourceProvider};
use std::path::PathBuf;

/// # In-memory raw buffer resource
#[derive(PartialEq, Eq, Clone)]
pub struct BufferResource<'a> {
    /// The data representing the underlying resource
    pub data: &'a [u8],
}

impl ResourceProvider for BufferResource<'_> {
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
    /// let weights_resource = BufferResource { data: &data };
    /// let weights = weights_resource.get_resource();
    /// ```
    fn get_resource(&self) -> Result<Resource, RustBertError> {
        Ok(Resource::Buffer(self.data))
    }
}

impl<'a> From<&'a Vec<u8>> for BufferResource<'a> {
    fn from(data: &'a Vec<u8>) -> Self {
        Self {
            data: data.as_slice(),
        }
    }
}

impl<'a> From<&'a Vec<u8>> for Box<dyn ResourceProvider + 'a> {
    fn from(data: &'a Vec<u8>) -> Self {
        Box::new(BufferResource {
            data: data.as_slice(),
        })
    }
}
