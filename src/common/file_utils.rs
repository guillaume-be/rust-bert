use crate::common::error::RustBertError;
use cached_path::Cache;
use lazy_static::lazy_static;
use std::env;
use std::path::PathBuf;

lazy_static! {
    pub static ref CACHE: Cache = Cache::builder()
        .dir(_get_cache_directory())
        .build()
        .unwrap();
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

pub fn cached_path(resource: &str) -> Result<PathBuf, RustBertError> {
    return Ok(CACHE.cached_path(resource)?);
}
