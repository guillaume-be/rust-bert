use lazy_static::lazy_static;
use std::path::PathBuf;
use reqwest::Client;
use std::fs;
use tokio::prelude::*;

extern crate dirs;

#[derive(PartialEq, Clone)]
pub enum Resource {
    Local(LocalResource),
    Remote(RemoteResource),
}

impl Resource {
    pub fn get_local_path(&self) -> &PathBuf {
        match self {
            Resource::Local(resource) => &resource.local_path,
            Resource::Remote(resource) => &resource.local_path,
        }
    }
}

#[derive(PartialEq, Clone)]
pub struct LocalResource {
    pub local_path: PathBuf
}

#[derive(PartialEq, Clone)]
pub struct RemoteResource {
    pub url: String,
    pub local_path: PathBuf,
}

impl RemoteResource {
    pub fn new(url: &str, target: PathBuf) -> RemoteResource {
        RemoteResource { url: url.to_string(), local_path: target }
    }

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
    pub static ref CACHE_DIRECTORY: PathBuf = _get_cache_directory();
}

fn _get_cache_directory() -> PathBuf {
    let mut home: PathBuf = dirs::home_dir().unwrap();
    home.push(".cache");
    home.push(".rustbert");
    home
}

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