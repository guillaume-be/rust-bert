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
//! resource location. Two types of resources exist:
//! - LocalResource: points to a local file
//! - RemoteResource: points to a remote file via a URL and a local cached file
//!
//! For both types of resources, the local location of teh file can be retrieved using
//! `get_local_path`, allowing to reference the resource file location regardless if it is a remote
//! or local resource. Default implementations for a number of `RemoteResources` are available as
//! pre-trained models in each model module.

use crate::common::error::RustBertError;
use cached_path::{Cache, Options, ProgressBar};
use lazy_static::lazy_static;
use std::env;
use std::path::PathBuf;

extern crate dirs;

/// # Resource Enum pointing to model, configuration or vocabulary resources
/// Can be of type:
/// - LocalResource
/// - RemoteResource
#[derive(PartialEq, Clone)]
pub enum Resource {
    Local(LocalResource),
    Remote(RemoteResource),
}

impl Resource {
    /// Gets the local path for a given resource.
    ///
    /// If the resource is a remote resource, it is downloaded and cached. Then the path
    /// to the local cache is returned.
    ///
    /// # Returns
    ///
    /// * `PathBuf` pointing to the resource file
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::resources::{LocalResource, Resource};
    /// use std::path::PathBuf;
    /// let config_resource = Resource::Local(LocalResource {
    ///     local_path: PathBuf::from("path/to/config.json"),
    /// });
    /// let config_path = config_resource.get_local_path();
    /// ```
    pub fn get_local_path(&self) -> Result<PathBuf, RustBertError> {
        match self {
            Resource::Local(resource) => Ok(resource.local_path.clone()),
            Resource::Remote(resource) => {
                let cached_path = CACHE.cached_path_with_options(
                    &resource.url,
                    &Options::default().subdir(&resource.cache_subdir),
                )?;
                Ok(cached_path)
            }
        }
    }
}

/// # Local resource
#[derive(PartialEq, Clone)]
pub struct LocalResource {
    /// Local path for the resource
    pub local_path: PathBuf,
}

/// # Remote resource
#[derive(PartialEq, Clone)]
pub struct RemoteResource {
    /// Remote path/url for the resource
    pub url: String,
    /// Local subdirectory of the cache root where this resource is saved
    pub cache_subdir: String,
}

impl RemoteResource {
    /// Creates a new RemoteResource from an URL and a custom local path. Note that this does not
    /// download the resource (only declares the remote and local locations)
    ///
    /// # Arguments
    ///
    /// * `url` - `&str` Location of the remote resource
    /// * `cache_subdir` - `&str` Local subdirectory of the cache root to save the resource to
    ///
    /// # Returns
    ///
    /// * `RemoteResource` RemoteResource object
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::resources::{RemoteResource, Resource};
    /// let config_resource = Resource::Remote(RemoteResource::new(
    ///     "configs",
    ///     "http://config_json_location",
    /// ));
    /// ```
    pub fn new(url: &str, cache_subdir: &str) -> RemoteResource {
        RemoteResource {
            url: url.to_string(),
            cache_subdir: cache_subdir.to_string(),
        }
    }

    /// Creates a new RemoteResource from an URL and local name. Will define a local path pointing to
    /// ~/.cache/.rustbert/model_name. Note that this does not download the resource (only declares
    /// the remote and local locations)
    ///
    /// # Arguments
    ///
    /// * `name_url_tuple` - `(&str, &str)` Location of the name of model and remote resource
    ///
    /// # Returns
    ///
    /// * `RemoteResource` RemoteResource object
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::resources::{RemoteResource, Resource};
    /// let model_resource = Resource::Remote(RemoteResource::from_pretrained((
    ///     "distilbert-sst2",
    ///     "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/rust_model.ot",
    /// )));
    /// ```
    pub fn from_pretrained(name_url_tuple: (&str, &str)) -> RemoteResource {
        let cache_subdir = name_url_tuple.0.to_string();
        let url = name_url_tuple.1.to_string();
        RemoteResource { url, cache_subdir }
    }
}

lazy_static! {
    #[derive(Copy, Clone, Debug)]
/// # Global cache directory
/// If the environment variable `RUSTBERT_CACHE` is set, will save the cache model files at that
/// location. Otherwise defaults to `~/.cache/.rustbert`.
    pub static ref CACHE: Cache = Cache::builder()
        .dir(_get_cache_directory())
        .progress_bar(Some(ProgressBar::Light))
        .build().unwrap();
}

fn _get_cache_directory() -> PathBuf {
    match env::var("RUSTBERT_CACHE") {
        Ok(value) => PathBuf::from(value),
        Err(_) => {
            let mut home = dirs::home_dir().unwrap();
            home.push(".cache");
            home.push(".rustbert");
            home
        }
    }
}

#[deprecated(
    since = "0.9.1",
    note = "Please use `Resource.get_local_path()` instead"
)]
/// # (Download) the resource and return a path to its local path
/// This function will download remote resource to their local path if they do not exist yet.
/// Then for both `LocalResource` and `RemoteResource`, it will the local path to the resource.
/// For `LocalResource` only the resource path is returned.
///
/// # Arguments
///
/// * `resource` - Pointer to the `&Resource` to optionally download and get the local path.
///
/// # Returns
///
/// * `&PathBuf` Local path for the resource
///
/// # Example
///
/// ```no_run
/// use rust_bert::resources::{RemoteResource, Resource};
/// let model_resource = Resource::Remote(RemoteResource::from_pretrained((
///     "distilbert-sst2/model.ot",
///     "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/rust_model.ot",
/// )));
/// let local_path = model_resource.get_local_path();
/// ```
pub fn download_resource(resource: &Resource) -> Result<PathBuf, RustBertError> {
    resource.get_local_path()
}
