use std::path::Path;
use lazy_static::lazy_static;
use std::path::PathBuf;
use reqwest::Client;
use std::fs;
use tokio::prelude::*;

extern crate dirs;

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
pub async fn download_file(url: &str, target: &Path) -> failure::Fallible<()> {
    fs::create_dir_all(target.parent().unwrap())?;

    let client = Client::new();
    let mut output_file = tokio::fs::File::create(target).await?;
    let mut response = client.get(url).send().await?;
    while let Some(chunk) = response.chunk().await? {
        output_file.write_all(&chunk).await?;
    }
    Ok(())
}
