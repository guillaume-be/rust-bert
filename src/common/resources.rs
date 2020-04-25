use lazy_static::lazy_static;
use std::path::PathBuf;
use reqwest::Client;
use std::fs;
use tokio::prelude::*;

extern crate dirs;

pub enum Dependency {
    Local(LocalDependency),
    Remote(RemoteDependency),
}

impl Dependency {
    pub fn get_local_path(&self) -> &PathBuf {
        match self {
            Dependency::Local(dependency) => &dependency.local_path,
            Dependency::Remote(dependency) => &dependency.local_path,
        }
    }
}

pub struct LocalDependency {
    pub local_path: PathBuf
}

pub struct RemoteDependency {
    pub url: String,
    pub local_path: PathBuf,
}

impl RemoteDependency {
    pub fn new(url: &str, target: PathBuf) -> RemoteDependency {
        RemoteDependency { url: url.to_string(), local_path: target }
    }

    pub fn from_pretrained(name_url_tuple: (&str, &str)) -> RemoteDependency {
        let name = name_url_tuple.0;
        let url = name_url_tuple.1.to_string();
        let mut local_path = CACHE_DIRECTORY.to_path_buf();
        local_path.push(name);
        RemoteDependency { url, local_path }
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
pub async fn download_dependency(dependency: Dependency) -> failure::Fallible<()> {
    match dependency {
        Dependency::Remote(dependency) => {
            let target = dependency.local_path;
            let url = dependency.url;
            if !target.exists() {
                fs::create_dir_all(target.parent().unwrap())?;

                let client = Client::new();
                let mut output_file = tokio::fs::File::create(target).await?;
                let mut response = client.get(url.as_str()).send().await?;
                while let Some(chunk) = response.chunk().await? {
                    output_file.write_all(&chunk).await?;
                }
            }
            Ok(())
        }
        Dependency::Local(_) => {
            Ok(())
        }
    }
}