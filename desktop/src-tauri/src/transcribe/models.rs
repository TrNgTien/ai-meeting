//! The GGML model cache: what is on disk, how big it is, and how to get more.
//!
//! Port of the model-cache half of `transcriber.py` (`is_model_downloaded`,
//! `model_size_on_disk`, `delete_model`, `list_downloaded_whisper_models`,
//! `ensure_model_downloaded`) and the backend for the Manage Models dialog.
//!
//! The checkpoints are different files than the Python app's: whisper.cpp wants
//! GGML, openai-whisper wanted `.pt`. They therefore live in a **different
//! directory**, and this module never touches `~/.cache/whisper` — the Python
//! app is still using it, and deleting its models out from under it would be a
//! nasty surprise while both apps are installed.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};

use crate::state::CancelFlag;

/// Where whisper.cpp GGML checkpoints are published.
const GGML_REPO_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// Bytes to move between the socket and the file at a time. Large enough that
/// the progress callback fires a few times a second on a fast link rather than
/// thousands.
const DOWNLOAD_CHUNK: usize = 1 << 20;

/// How long to wait for a connection, and then for response headers.
///
/// Both matter: Hugging Face redirects to a CDN that occasionally accepts the
/// connection and then never sends headers. Without a timeout the app hangs
/// forever on a model download with no way out but Force Quit — observed, not
/// hypothetical.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(60);

/// Consecutive attempts that fetch *nothing* before we give up.
///
/// Attempts that make progress don't count against this: Hugging Face's CDN
/// drops these multi-gigabyte transfers regularly, and since each retry resumes
/// with a Range request, a run that keeps gaining ground should be allowed to
/// finish. Only a stretch of attempts that all fail to move a single byte means
/// something is actually wrong.
const FRUITLESS_ATTEMPTS: usize = 6;

/// Absolute ceiling, so a pathological server cannot spin here forever.
const MAX_ATTEMPTS: usize = 60;

fn agent() -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::Agent::config_builder()
            .timeout_connect(Some(CONNECT_TIMEOUT))
            .timeout_recv_response(Some(RESPONSE_TIMEOUT))
            .build(),
    )
}

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    /// The user pressed Cancel. The partial file is removed.
    #[error("download cancelled")]
    Cancelled,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// `(model_name, downloaded_bytes, total_bytes)`. `total` is 0 when the server
/// does not report a Content-Length.
pub type ProgressCallback<'a> = dyn Fn(&str, u64, u64) + Send + Sync + 'a;

/// GGML filename for a model name from [`super::MODEL_OPTIONS`].
pub fn ggml_filename(name: &str) -> String {
    format!("ggml-{name}.bin")
}

/// `$XDG_CACHE_HOME/whisper-cpp`, else `~/.cache/whisper-cpp`.
///
/// Deliberately *not* `~/.cache/whisper`: see the module docs.
pub fn cache_dir() -> PathBuf {
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(xdg).join("whisper-cpp");
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("whisper-cpp")
}

pub fn model_path(name: &str) -> PathBuf {
    cache_dir().join(ggml_filename(name))
}

/// Where an in-flight download is written. Renamed into place only when
/// complete, so a cancelled or crashed download can never be mistaken for a
/// usable model.
fn partial_path(name: &str) -> PathBuf {
    cache_dir().join(format!("{}.part", ggml_filename(name)))
}

pub fn is_model_downloaded(name: &str) -> bool {
    model_path(name).is_file()
}

pub fn model_size_on_disk(name: &str) -> u64 {
    fs::metadata(model_path(name)).map(|m| m.len()).unwrap_or(0)
}

/// Delete one model's file, freeing that disk space only.
///
/// Returns whether anything was removed. The model re-downloads automatically
/// the next time it is selected and used.
pub fn delete_model(name: &str) -> bool {
    let path = model_path(name);
    if !path.is_file() {
        return false;
    }
    fs::remove_file(&path).is_ok()
}

/// The models from [`super::MODEL_OPTIONS`] that are actually on disk, in the
/// same fastest-to-most-accurate order the dropdown shows.
///
/// The header dropdown is built from this so that picking a model never stalls
/// on a multi-gigabyte download.
pub fn list_downloaded_models() -> Vec<String> {
    super::MODEL_OPTIONS
        .iter()
        .filter(|name| is_model_downloaded(name))
        .map(|name| name.to_string())
        .collect()
}

/// Size of a model on the server, for showing "1.5 GB" before committing.
///
/// `None` when the server does not say; the UI should then show the download as
/// indeterminate rather than inventing a number.
pub fn remote_size(name: &str) -> Option<u64> {
    let url = format!("{GGML_REPO_BASE}/{}", ggml_filename(name));
    let response = agent().head(&url).call().ok()?;
    header_u64(response.headers(), "content-length")
}

fn header_u64(headers: &ureq::http::HeaderMap, name: &str) -> Option<u64> {
    headers.get(name)?.to_str().ok()?.parse().ok()
}

/// Total size out of a `Content-Range: bytes 100-1599/1600` header.
fn total_from_content_range(headers: &ureq::http::HeaderMap) -> Option<u64> {
    headers
        .get("content-range")?
        .to_str()
        .ok()?
        .rsplit('/')
        .next()?
        .trim()
        .parse()
        .ok()
}

/// Make sure `name` is on disk, downloading it if not, and return its path.
///
/// Reports progress as it goes and honours `cancel` between chunks, so a user
/// who started a 1.5 GB download by accident is not stuck waiting for it.
pub fn ensure_model_downloaded(
    name: &str,
    progress: Option<&ProgressCallback<'_>>,
    cancel: Option<&CancelFlag>,
) -> Result<PathBuf, DownloadError> {
    let final_path = model_path(name);
    if final_path.is_file() {
        return Ok(final_path);
    }
    if !super::MODEL_OPTIONS.contains(&name) {
        return Err(DownloadError::Other(anyhow!("unknown model '{name}'")));
    }

    fs::create_dir_all(cache_dir())
        .with_context(|| format!("cannot create {}", cache_dir().display()))
        .map_err(DownloadError::Other)?;

    let partial = partial_path(name);
    let partial_len = || fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);

    let mut last_error = None;
    let mut fruitless = 0usize;

    for attempt in 1..=MAX_ATTEMPTS {
        let before = partial_len();

        match fetch_into_partial(name, &partial, progress, cancel) {
            Ok(()) => {
                fs::rename(&partial, &final_path)
                    .with_context(|| format!("cannot move {} into place", partial.display()))
                    .map_err(DownloadError::Other)?;
                return Ok(final_path);
            }
            Err(DownloadError::Cancelled) => {
                // Cancelling deletes the partial file, as the Python app's
                // DownloadCancelled path did: a half-model is only confusing.
                let _ = fs::remove_file(&partial);
                return Err(DownloadError::Cancelled);
            }
            Err(error) => {
                last_error = Some(error);
                if partial_len() > before {
                    // Ground gained. Keep going for as long as that holds.
                    fruitless = 0;
                } else {
                    fruitless += 1;
                }
                if fruitless >= FRUITLESS_ATTEMPTS || attempt == MAX_ATTEMPTS {
                    break;
                }
                std::thread::sleep(Duration::from_secs(2 * fruitless.max(1) as u64));
            }
        }
    }

    // The partial file is deliberately *kept*: it is a resumable cache, and
    // discarding a gigabyte of successfully transferred data because the last
    // stretch failed is the wrong trade. The next attempt — this run or the next
    // launch of the app — picks up where this left off.
    let kept = partial_len();
    Err(match last_error {
        Some(DownloadError::Other(error)) if kept > 0 => DownloadError::Other(error.context(
            format!(
                "downloaded {} MB of '{name}' before failing; it is kept and \
                 will resume on the next attempt",
                kept / 1_000_000
            ),
        )),
        Some(error) => error,
        None => DownloadError::Other(anyhow!("download failed")),
    })
}

/// One download attempt, resuming from `partial` if it already has bytes.
fn fetch_into_partial(
    name: &str,
    partial: &Path,
    progress: Option<&ProgressCallback<'_>>,
    cancel: Option<&CancelFlag>,
) -> Result<(), DownloadError> {
    let url = format!("{GGML_REPO_BASE}/{}", ggml_filename(name));
    let mut already = fs::metadata(partial).map(|m| m.len()).unwrap_or(0);

    let mut request = agent().get(&url);
    if already > 0 {
        request = request.header("Range", format!("bytes={already}-"));
    }

    let response = request
        .call()
        .with_context(|| format!("cannot fetch {url}"))
        .map_err(DownloadError::Other)?;

    // 206 means the server honoured the Range. Anything else (a 200) means it
    // sent the whole file, so the bytes we already have would be duplicated.
    let resuming = response.status().as_u16() == 206;
    if already > 0 && !resuming {
        let _ = fs::remove_file(partial);
        already = 0;
    }

    let total = if resuming {
        total_from_content_range(response.headers()).unwrap_or(0)
    } else {
        header_u64(response.headers(), "content-length").unwrap_or(0)
    };

    let file = if resuming {
        OpenOptions::new()
            .append(true)
            .open(partial)
            .with_context(|| format!("cannot resume {}", partial.display()))
            .map_err(DownloadError::Other)?
    } else {
        File::create(partial)
            .with_context(|| format!("cannot create {}", partial.display()))
            .map_err(DownloadError::Other)?
    };

    stream_to_file(
        file,
        response.into_body().into_reader(),
        name,
        already,
        total,
        progress,
        cancel,
    )
}

/// Copy the response body into `file`, reporting progress and honouring cancel.
///
/// `already` is how many bytes a resumed download starts with, so the progress
/// numbers describe the file rather than this attempt.
fn stream_to_file(
    mut file: File,
    mut reader: impl Read,
    name: &str,
    already: u64,
    total: u64,
    progress: Option<&ProgressCallback<'_>>,
    cancel: Option<&CancelFlag>,
) -> Result<(), DownloadError> {
    let mut buffer = vec![0u8; DOWNLOAD_CHUNK];
    let mut downloaded: u64 = already;

    loop {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(DownloadError::Cancelled);
        }
        let read = reader
            .read(&mut buffer)
            .context("connection failed mid-download")
            .map_err(DownloadError::Other)?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])
            .context("cannot write to the model cache")
            .map_err(DownloadError::Other)?;
        downloaded += read as u64;
        if let Some(report) = progress {
            report(name, downloaded, total);
        }
    }

    file.flush().map_err(|e| DownloadError::Other(e.into()))?;
    file.sync_all()
        .map_err(|e| DownloadError::Other(e.into()))?;

    if total > 0 && downloaded != total {
        return Err(DownloadError::Other(anyhow!(
            "download truncated: got {downloaded} of {total} bytes"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ggml_names_match_the_published_files() {
        assert_eq!(ggml_filename("large-v3"), "ggml-large-v3.bin");
        assert_eq!(ggml_filename("large-v3-turbo"), "ggml-large-v3-turbo.bin");
        assert_eq!(ggml_filename("small"), "ggml-small.bin");
    }

    #[test]
    fn the_cache_is_not_the_python_apps_cache() {
        // Sharing the directory would let this app's Manage Models dialog
        // delete checkpoints the Python app is still using.
        let dir = cache_dir();
        assert!(dir.ends_with("whisper-cpp"), "{}", dir.display());
        assert!(!dir.ends_with("whisper"));
    }

    #[test]
    fn xdg_cache_home_is_honoured() {
        // Serialised with the other env-var-free tests by construction: this is
        // the only test that touches XDG_CACHE_HOME, and it restores it.
        let previous = std::env::var_os("XDG_CACHE_HOME");
        // SAFETY: single-threaded within this test, and restored below.
        unsafe { std::env::set_var("XDG_CACHE_HOME", "/tmp/xdg-example") };
        assert_eq!(cache_dir(), PathBuf::from("/tmp/xdg-example/whisper-cpp"));
        match previous {
            Some(value) => unsafe { std::env::set_var("XDG_CACHE_HOME", value) },
            None => unsafe { std::env::remove_var("XDG_CACHE_HOME") },
        }
    }

    #[test]
    fn unknown_models_are_refused_rather_than_fetched() {
        let error = ensure_model_downloaded("not-a-model", None, None).unwrap_err();
        assert!(matches!(error, DownloadError::Other(_)));
        assert!(error.to_string().contains("unknown model"));
    }

    #[test]
    fn a_cancelled_download_reports_cancellation() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let partial = dir.path().join("ggml-small.bin.part");
        let cancel = CancelFlag::new();
        cancel.cancel();

        let data = vec![0u8; 4096];
        let error = stream_to_file(
            File::create(&partial)?,
            data.as_slice(),
            "small",
            0,
            4096,
            None,
            Some(&cancel),
        )
        .unwrap_err();

        assert!(matches!(error, DownloadError::Cancelled));
        Ok(())
    }

    #[test]
    fn a_truncated_download_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let partial = dir.path().join("ggml-small.bin.part");
        let data = vec![0u8; 100];
        // Claim more than we deliver, as a dropped connection would.
        let error = stream_to_file(
            File::create(&partial)?,
            data.as_slice(),
            "small",
            0,
            999_999,
            None,
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("truncated"));
        Ok(())
    }

    #[test]
    fn progress_is_reported_with_the_model_name() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let partial = dir.path().join("ggml-small.bin.part");
        let seen = std::sync::Mutex::new(Vec::new());
        let report = |name: &str, done: u64, total: u64| {
            seen.lock().unwrap().push((name.to_string(), done, total));
        };

        let data = vec![7u8; DOWNLOAD_CHUNK + 512];
        stream_to_file(
            File::create(&partial)?,
            data.as_slice(),
            "small",
            0,
            data.len() as u64,
            Some(&report),
            None,
        )?;

        let seen = seen.lock().unwrap();
        assert!(seen.len() >= 2, "expected several progress ticks: {seen:?}");
        assert_eq!(seen.last().unwrap().1, data.len() as u64);
        assert!(seen.iter().all(|(name, _, _)| name == "small"));
        assert_eq!(fs::metadata(&partial)?.len(), data.len() as u64);
        Ok(())
    }

    #[test]
    fn a_resumed_download_counts_the_bytes_already_on_disk() -> Result<(), Box<dyn std::error::Error>>
    {
        let dir = tempfile::tempdir()?;
        let partial = dir.path().join("ggml-small.bin.part");
        fs::write(&partial, vec![1u8; 600])?;

        let seen = std::sync::Mutex::new(Vec::new());
        let report = |_: &str, done: u64, total: u64| seen.lock().unwrap().push((done, total));

        // The server sends the remaining 400 bytes of a 1000-byte file.
        stream_to_file(
            OpenOptions::new().append(true).open(&partial)?,
            vec![2u8; 400].as_slice(),
            "small",
            600,
            1000,
            Some(&report),
            None,
        )?;

        assert_eq!(fs::metadata(&partial)?.len(), 1000, "appended, not overwritten");
        assert_eq!(seen.lock().unwrap().last().unwrap(), &(1000, 1000));
        Ok(())
    }

    #[test]
    fn content_range_yields_the_total_size() {
        let mut headers = ureq::http::HeaderMap::new();
        headers.insert("content-range", "bytes 600-999/1000".parse().unwrap());
        assert_eq!(total_from_content_range(&headers), Some(1000));

        // An unsatisfiable-range reply has no total to extract.
        let mut starless = ureq::http::HeaderMap::new();
        starless.insert("content-range", "bytes */1000".parse().unwrap());
        assert_eq!(total_from_content_range(&starless), Some(1000));

        assert_eq!(total_from_content_range(&ureq::http::HeaderMap::new()), None);
    }
}
