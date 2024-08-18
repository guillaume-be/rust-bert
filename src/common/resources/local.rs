use crate::common::error::RustBertError;
use crate::resources::{Resource, ResourceProvider};
use std::path::PathBuf;

/// # Local resource
#[derive(PartialEq, Eq, Debug, Clone)]
pub struct LocalResource {
    /// Local path for the resource
    pub local_path: PathBuf,
}

impl ResourceProvider for LocalResource {
    /// Gets the path for a local resource.
    ///
    /// # Returns
    ///
    /// * `PathBuf` pointing to the resource file
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::resources::{LocalResource, ResourceProvider};
    /// use std::path::PathBuf;
    /// let config_resource = LocalResource {
    ///     local_path: PathBuf::from("path/to/config.json"),
    /// };
    /// let config_path = config_resource.get_local_path();
    /// ```
    fn get_local_path(&self) -> Result<PathBuf, RustBertError> {
        Ok(self.local_path.clone())
    }

    /// Gets a wrapper around the path for a local resource.
    ///
    /// # Returns
    ///
    /// * `Resource` wrapping a `PathBuf` pointing to the resource file
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::resources::{LocalResource, ResourceProvider};
    /// use std::path::PathBuf;
    /// let config_resource = LocalResource {
    ///     local_path: PathBuf::from("path/to/config.json"),
    /// };
    /// let config_path = config_resource.get_resource();
    /// ```
    fn get_resource(&self) -> Result<Resource, RustBertError> {
        Ok(Resource::PathBuf(self.local_path.clone()))
    }
}

impl From<PathBuf> for LocalResource {
    fn from(local_path: PathBuf) -> Self {
        Self { local_path }
    }
}

impl From<PathBuf> for Box<dyn ResourceProvider + Send> {
    fn from(local_path: PathBuf) -> Self {
        Box::new(LocalResource { local_path })
    }
}
