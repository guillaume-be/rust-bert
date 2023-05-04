//! # Resource definitions for model weights, vocabularies and configuration files
//!
//! This crate relies on the concept of Resources to access the data used by the models.
//! This includes:
//! - model weights
//! - configuration files
//! - vocabularies
//! - (optional) merges files for BPE-based tokenizers
//!
//! These are expected in the pipelines configurations or are used as utilities to reference to the
//! resource location. Two types of resources are pre-defined:
//! - LocalResource: points to a local file
//! - RemoteResource: points to a remote file via a URL
//! - TensorResource: refers to name/tensor mappings expected by tch
//! - BufferResource: refers to a buffer that contains file contents for a resource (currently only
//!                   usable for weights)
//!
//! For `LocalResource` and `RemoteResource`, the local location of the file can be retrieved using
//! `get_local_path`, allowing to reference the resource file location regardless if it is a remote
//! or local resource. Default implementations for a number of `RemoteResources` are available as
//! pre-trained models in each model module.

mod buffer;
mod local;
mod named_tensors;

use crate::common::error::RustBertError;
pub use buffer::BufferResource;
pub use local::LocalResource;
pub use named_tensors::TensorResource;
use std::{collections::HashMap, path::PathBuf, sync::RwLockReadGuard};
use tch::{nn::VarStore, Tensor};

pub enum Resource<'a> {
    PathBuf(PathBuf),
    Buffer(&'a Vec<u8>),
    NamedTensors(RwLockReadGuard<'a, HashMap<String, Tensor>>),
}

/// # Resource Trait that can provide the location or data for the model, and location of
/// configuration or vocabulary resources
pub trait ResourceProvider: Send + Sync {
    /// Provides the local path for a resource.
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
    fn get_local_path(&self) -> Result<PathBuf, RustBertError>;

    /// Provides access to an underlying resource.
    ///
    /// # Returns
    ///
    /// * `Resource` wrapping a representation of a resource.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::resources::{LocalResource, ResourceProvider, TensorResource};
    ///
    ///
    /// ```
    fn get_resource(&self) -> Result<Resource, RustBertError>;
}

/// Load the provided `VarStore` with named tensor values from the provided `ResourceProvider`
pub fn load_weights(
    rp: &(impl ResourceProvider + ?Sized),
    vs: &mut VarStore,
) -> Result<(), RustBertError> {
    match rp.get_resource()? {
        Resource::Buffer(data) => Ok(vs.load_from_stream(std::io::Cursor::new(&data))?),
        Resource::PathBuf(path) => Ok(vs.load(path)?),
        Resource::NamedTensors(named_tensors) => named_tensors::load_weights(named_tensors, vs),
    }
}

#[cfg(feature = "remote")]
mod remote;
#[cfg(feature = "remote")]
pub use remote::RemoteResource;
