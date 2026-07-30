//! Getting 16 kHz mono float32 out of an audio file, a range at a time.
//!
//! Port of `chunking.decode_range` / `chunking.audio_duration`, still via
//! ffmpeg and deliberately so:
//!
//! * Decoding `[start, start+duration)` with `-ss`/`-t` keeps memory flat
//!   regardless of recording length, which is the whole reason a four-hour file
//!   works at all.
//! * ffmpeg covers every format the app advertises. The pure-Rust decoders
//!   (symphonia) still have no opus or wma, so dropping ffmpeg would quietly
//!   shrink the supported list.
//! * It decodes and resamples *identically to the Python app*, which is what
//!   makes an A/B transcript diff meaningful: any difference is the model, not
//!   the samples fed to it.

use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, Context, Result};

use crate::SAMPLE_RATE;

/// Decode `[start_sec, start_sec + duration_sec)` as mono 16 kHz float32.
pub fn decode_range(source: &Path, start_sec: f64, duration_sec: f64) -> Result<Vec<f32>> {
    let output = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-threads",
            "0",
            "-ss",
            &format!("{:.3}", start_sec.max(0.0)),
            "-t",
            &format!("{:.3}", duration_sec.max(0.0)),
            "-i",
        ])
        .arg(source)
        .args([
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-",
        ])
        .output()
        .with_context(|| {
            "ffmpeg could not be run. Install it with `brew install ffmpeg` \
             (macOS) or `apt-get install ffmpeg` (Debian/Ubuntu)."
        })?;

    if !output.status.success() {
        // ffmpeg's last stderr line is the one that says what went wrong; the
        // rest is banner noise.
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = stderr
            .lines()
            .rfind(|line| !line.trim().is_empty())
            .unwrap_or("unknown ffmpeg error");
        let name = source.file_name().unwrap_or(source.as_os_str());
        return Err(anyhow!(
            "ffmpeg failed to decode {}: {detail}",
            name.to_string_lossy()
        ));
    }

    Ok(pcm_s16le_to_f32(&output.stdout))
}

/// Interpret raw little-endian signed 16-bit PCM as float32 in [-1, 1).
///
/// Divides by 32768 rather than 32767, matching the Python app so the two
/// produce bit-identical input for the same file.
fn pcm_s16le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32768.0)
        .collect()
}

/// Duration in seconds via ffprobe, or `None` if it can't be determined.
///
/// `None` is not fatal: the chunk loop also stops when a decode comes back
/// shorter than requested, which is what happens at the end of the file.
pub fn audio_duration(source: &Path) -> Option<f64> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
        ])
        .arg(source)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }
    let value: f64 = String::from_utf8_lossy(&output.stdout).trim().parse().ok()?;
    (value > 0.0).then_some(value)
}

/// Whether ffmpeg is on PATH, so the UI can say so up front instead of failing
/// on the first chunk.
pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_conversion_matches_the_python_scaling() {
        // 0, +1, -1, and full negative scale.
        let bytes = [0x00, 0x00, 0x01, 0x00, 0xff, 0xff, 0x00, 0x80];
        let samples = pcm_s16le_to_f32(&bytes);
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0], 0.0);
        assert_eq!(samples[1], 1.0 / 32768.0);
        assert_eq!(samples[2], -1.0 / 32768.0);
        assert_eq!(samples[3], -1.0);
    }

    #[test]
    fn a_trailing_odd_byte_is_ignored_rather_than_panicking() {
        let samples = pcm_s16le_to_f32(&[0x00, 0x00, 0x7f]);
        assert_eq!(samples.len(), 1);
    }
}
