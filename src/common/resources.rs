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

use crate::common::error::RustBertError;
use std::path::PathBuf;

/// # Resource Trait that can provide the location of the model, configuration or vocabulary resources
pub trait ResourceProvider {
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
}

/// # Local resource
#[derive(PartialEq, Clone)]
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
}

#[cfg(feature = "remote")]
pub mod remote {
    use cached_path::{Cache, Options, ProgressBar};
    use dirs::cache_dir;
    use lazy_static::lazy_static;

    use super::*;

    /// # Remote resource that will be downloaded and cached locally on demand
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
        /// use rust_bert::resources::remote::RemoteResource;
        /// let config_resource = RemoteResource::new(
        ///     "configs",
        ///     "http://config_json_location",
        /// );
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
        /// use rust_bert::resources::remote::RemoteResource;
        /// let model_resource = RemoteResource::from_pretrained((
        ///     "distilbert-sst2",
        ///     "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/rust_model.ot",
        /// ));
        /// ```
        pub fn from_pretrained(name_url_tuple: (&str, &str)) -> RemoteResource {
            let cache_subdir = name_url_tuple.0.to_string();
            let url = name_url_tuple.1.to_string();
            RemoteResource { url, cache_subdir }
        }
    }

    impl ResourceProvider for RemoteResource {
        /// Gets the local path for a remote resource.
        ///
        /// The remote resource is downloaded and cached. Then the path
        /// to the local cache is returned.
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
            let cached_path = CACHE.cached_path_with_options(
                &self.url,
                &Options::default().subdir(&self.cache_subdir),
            )?;
            Ok(cached_path)
        }
    }

    lazy_static! {
        #[derive(Copy, Clone, Debug)]
    /// # Global cache directory
    /// If the environment variable `RUSTBERT_CACHE` is set, will save the cache model files at that
    /// location. Otherwise defaults to `$XDG_CACHE_HOME/.rustbert`, or corresponding user cache for
    /// the current system.
        pub static ref CACHE: Cache = Cache::builder()
            .dir(_get_cache_directory())
            .progress_bar(Some(ProgressBar::Light))
            .build().unwrap();
    }

    fn _get_cache_directory() -> PathBuf {
        match std::env::var("RUSTBERT_CACHE") {
            Ok(value) => PathBuf::from(value),
            Err(_) => {
                let mut home = cache_dir().unwrap();
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
    /// use rust_bert::resources::{remote::RemoteResource, ResourceProvider};
    /// let model_resource = RemoteResource::from_pretrained((
    ///     "distilbert-sst2/model.ot",
    ///     "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/rust_model.ot",
    /// ));
    /// let local_path = model_resource.get_local_path();
    /// ```
    pub fn download_resource(resource: &dyn ResourceProvider) -> Result<PathBuf, RustBertError> {
        resource.get_local_path()
    }
}
