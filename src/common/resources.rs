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

use lazy_static::lazy_static;
use std::path::PathBuf;
use reqwest::Client;
use std::{fs, env};
use tokio::prelude::*;

extern crate dirs;

/// # Resource Enum expected by the `download_resource` function
/// Can be of type:
/// - LocalResource
/// - RemoteResource
#[derive(PartialEq, Clone)]
pub enum Resource {
    Local(LocalResource),
    Remote(RemoteResource),
}

impl Resource {
    /// Gets the local path for a given resource
    ///
    /// # Returns
    ///
    /// * `PathBuf` pointing to the resource file
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::resources::{Resource, LocalResource};
    /// use std::path::PathBuf;
    /// let config_resource = Resource::Local(LocalResource { local_path: PathBuf::from("path/to/config.json")});
    /// let config_path = config_resource.get_local_path();
    /// ```
    ///
    pub fn get_local_path(&self) -> &PathBuf {
        match self {
            Resource::Local(resource) => &resource.local_path,
            Resource::Remote(resource) => &resource.local_path,
        }
    }
}

/// # Local resource
#[derive(PartialEq, Clone)]
pub struct LocalResource {
    /// Local path for the resource
    pub local_path: PathBuf
}

/// # Remote resource
#[derive(PartialEq, Clone)]
pub struct RemoteResource {
    /// Remote path/url for the resource
    pub url: String,
    /// Local path for the resource
    pub local_path: PathBuf,
}

impl RemoteResource {
    /// Creates a new RemoteResource from an URL and a custom local path. Note that this does not
    /// download the resource (only declares the remote and local locations)
    ///
    /// # Arguments
    ///
    /// * `url` - `&str` Location of the remote resource
    /// * `target` - `PathBuf` Local path to save teh resource to
    ///
    /// # Returns
    ///
    /// * `RemoteResource` RemoteResource object
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rust_bert::resources::{Resource, RemoteResource};
    /// use std::path::PathBuf;
    /// let config_resource = Resource::Remote(RemoteResource::new("http://config_json_location", PathBuf::from("path/to/config.json")));
    /// ```
    ///
    pub fn new(url: &str, target: PathBuf) -> RemoteResource {
        RemoteResource { url: url.to_string(), local_path: target }
    }

    /// Creates a new RemoteResource from an URL and local name. Will define a local path pointing to
    /// ~/.cache/.rusbert/model_name. Note that this does not download the resource (only declares
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
    /// use rust_bert::resources::{Resource, RemoteResource};
    /// let model_resource = Resource::Remote(RemoteResource::from_pretrained(
    ///     ("distilbert-sst2/model.ot",
    ///     "https://cdn.huggingface.co/distilbert-base-uncased-finetuned-sst-2-english-rust_model.ot"
    ///     )
    /// ));
    /// ```
    ///
    pub fn from_pretrained(name_url_tuple: (&str, &str)) -> RemoteResource {
        let name = name_url_tuple.0;
        let url = name_url_tuple.1.to_string();
        let mut local_path = CACHE_DIRECTORY.to_path_buf();
        local_path.push(name);
        RemoteResource { url, local_path }
    }
}

lazy_static! {
    #[derive(Copy, Clone, Debug)]
/// # Global cache directory
/// If the environment variable `RUSTBERT_CACHE` is set, will save the cache model files at that
/// location. Otherwise defaults to `~/.cache/.rustbert`.
    pub static ref CACHE_DIRECTORY: PathBuf = _get_cache_directory();
}

fn _get_cache_directory() -> PathBuf {
    let home = match env::var("RUSTBERT_CACHE") {
        Ok(value) => PathBuf::from(value),
        Err(_) => {
            let mut home = dirs::home_dir().unwrap();
            home.push(".cache");
            home.push(".rustbert");
            home
        }
    };
    home
}

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
/// use rust_bert::resources::{Resource, RemoteResource, download_resource};
/// let model_resource = Resource::Remote(RemoteResource::from_pretrained(
///     ("distilbert-sst2/model.ot",
///     "https://cdn.huggingface.co/distilbert-base-uncased-finetuned-sst-2-english-rust_model.ot"
///     )
/// ));
/// let local_path = download_resource(&model_resource);
/// ```
///
#[tokio::main]
pub async fn download_resource(resource: &Resource) -> failure::Fallible<&PathBuf> {
    match resource {
        Resource::Remote(remote_resource) => {
            let target = &remote_resource.local_path;
            let url = &remote_resource.url;
            if !target.exists() {
                println!("Downloading {} to {:?}", url, target);
                fs::create_dir_all(target.parent().unwrap())?;

                let client = Client::new();
                let mut output_file = tokio::fs::File::create(target).await?;
                let mut response = client.get(url.as_str()).send().await?;
                while let Some(chunk) = response.chunk().await? {
                    output_file.write_all(&chunk).await?;
                }
            }
            Ok(resource.get_local_path())
        }
        Resource::Local(_) => {
            Ok(resource.get_local_path())
        }
    }
}