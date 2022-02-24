//! # Resource definitions for model weights, vocabularies and configuration files
//!
//! This crate relies on the concept of Resources to access the files used by the models.
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
//!
//! For both types of resources, the local location of the file can be retrieved using
//! `get_local_path`, allowing to reference the resource file location regardless if it is a remote
//! or local resource. Default implementations for a number of `RemoteResources` are available as
//! pre-trained models in each model module.

mod local;
#[cfg(feature = "remote")]
mod remote;

pub use local::{LocalResource, ResourceProvider};

#[cfg(feature = "remote")]
pub use remote::RemoteResource;
