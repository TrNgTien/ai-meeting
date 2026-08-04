//! Appending mono 16 kHz audio to a WAV, pinned to wall-clock time.
//!
//! Port of `audio_capture._MonoWavWriter`, and the single most load-bearing
//! piece of the recording path.
//!
//! The naive thing — write every buffer end to end — quietly desynchronises the
//! two tracks. The microphone runs on its device's clock, the system mix on
//! CoreAudio's, neither is exactly 16000 Hz, and either side can drop a buffer
//! when the machine is busy transcribing. Each of those shifts everything after
//! it, and the drift shows up as the "Me" and "Meeting" lines sliding apart over
//! a long meeting — exactly the thing the merged transcript depends on.
//!
//! So *position, not arrival order*, decides where audio lands: each block is
//! written where the wall clock says it belongs, with silence padding a gap and
//! frames dropped from a block that would run ahead. Both tracks share one `t0`,
//! so both stay locked to the same timeline and therefore to each other.

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{Context, Result};

use crate::SAMPLE_RATE;

/// How far a track may drift from wall-clock time before the writer corrects it.
/// Small enough that the two transcripts stay aligned to well within a spoken
/// word, large enough that ordinary callback jitter isn't constantly "corrected".
const RESYNC_TOLERANCE_SEC: f64 = 0.10;

const WAV_HEADER_LEN: u64 = 44;

/// A 16-bit PCM mono WAV being written incrementally.
///
/// Hand-rolled rather than pulled from a crate because the header has to be
/// rewritten on close with the final length, and because a recording that is
/// killed mid-take should still leave a file whose header is only wrong about
/// its length — most players cope, and ffmpeg certainly does.
pub struct MonoWavWriter {
    file: BufWriter<File>,
    closed: bool,
    /// The instant that becomes 00:00:00 in this track. `None` until both sides
    /// are actually running.
    t0: Option<f64>,
    frames: u64,
    peak: f32,
    drift_frames: i64,
}

impl MonoWavWriter {
    pub fn create(path: &Path) -> Result<Self> {
        let mut file = File::create(path)
            .with_context(|| format!("cannot create {}", path.display()))?;
        file.write_all(&wav_header(0))?;
        Ok(Self {
            file: BufWriter::new(file),
            closed: false,
            t0: None,
            frames: 0,
            peak: 0.0,
            drift_frames: 0,
        })
    }

    /// Declare the instant that becomes 00:00:00 in this track.
    ///
    /// Set once both sides are actually running, never at construction: a
    /// first-run permission prompt can hold a backend open for as long as the
    /// user takes to click it, and a `t0` chosen before that would bake the whole
    /// wait into the file as silence.
    pub fn set_origin(&mut self, t0: f64) {
        self.t0 = Some(t0);
    }

    /// Write one block. `started_at` is when its first sample was captured.
    pub fn append(&mut self, mono: &[f32], started_at: f64) -> Result<()> {
        if let Some(loudest) = mono.iter().map(|s| s.abs()).fold(None, max_option) {
            self.peak = self.peak.max(loudest);
        }

        let Some(t0) = self.t0 else {
            // Pre-roll: audio from a backend that started before the other side
            // was ready. It predates the shared timeline, so it has no position
            // on it.
            return Ok(());
        };
        if self.closed {
            return Ok(());
        }

        let expected = ((started_at - t0).max(0.0) * SAMPLE_RATE as f64) as i64;
        let gap = expected - self.frames as i64;
        let tolerance = (RESYNC_TOLERANCE_SEC * SAMPLE_RATE as f64) as i64;

        let mut block = mono;
        if gap > tolerance {
            // Late, or a buffer went missing: hold the timeline open with silence
            // so what follows keeps its true offset.
            self.write_silence(gap as u64)?;
            self.drift_frames += gap;
        } else if gap < -tolerance {
            // Running ahead of the clock; drop the overlap rather than push every
            // later word further out of place.
            let trim = ((-gap) as usize).min(block.len());
            block = &block[trim..];
            self.drift_frames -= trim as i64;
        }

        if !block.is_empty() {
            self.write_samples(block)?;
        }
        Ok(())
    }

    fn write_samples(&mut self, block: &[f32]) -> Result<()> {
        let mut bytes = Vec::with_capacity(block.len() * 2);
        for sample in block {
            bytes.extend_from_slice(&to_i16(*sample).to_le_bytes());
        }
        self.file.write_all(&bytes)?;
        self.frames += block.len() as u64;
        Ok(())
    }

    fn write_silence(&mut self, frames: u64) -> Result<()> {
        // Written in bounded batches: a long stall could otherwise ask for a
        // multi-megabyte allocation in one go.
        const BATCH: u64 = 16_384;
        let mut left = frames;
        let zeros = vec![0u8; (BATCH * 2) as usize];
        while left > 0 {
            let now = left.min(BATCH);
            self.file.write_all(&zeros[..(now * 2) as usize])?;
            left -= now;
        }
        self.frames += frames;
        Ok(())
    }

    /// Finish the file, patching the header with the real length. Returns frames
    /// written.
    pub fn close(&mut self) -> Result<u64> {
        if self.closed {
            return Ok(self.frames);
        }
        self.closed = true;
        self.file.flush()?;

        let data_len = self.frames * 2;
        let file = self.file.get_mut();
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&wav_header(data_len as u32))?;
        file.flush()?;
        file.sync_all()?;
        Ok(self.frames)
    }

    pub fn seconds(&self) -> f64 {
        self.frames as f64 / SAMPLE_RATE as f64
    }

    /// Signed silence/trim applied so far, as a health signal for the UI.
    pub fn drift_seconds(&self) -> f64 {
        self.drift_frames as f64 / SAMPLE_RATE as f64
    }

    /// Loudest sample since the last call — drives the level meter.
    pub fn take_peak(&mut self) -> f32 {
        std::mem::replace(&mut self.peak, 0.0)
    }
}

fn max_option(current: Option<f32>, next: f32) -> Option<f32> {
    Some(match current {
        Some(value) => value.max(next),
        None => next,
    })
}

/// Clamp before casting: a backend can hand us samples slightly outside
/// [-1, 1], and letting those wrap turns a loud moment into a burst of noise.
fn to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

/// A 44-byte canonical PCM WAV header for mono 16-bit at [`SAMPLE_RATE`].
fn wav_header(data_len: u32) -> [u8; WAV_HEADER_LEN as usize] {
    let mut header = [0u8; WAV_HEADER_LEN as usize];
    let byte_rate = SAMPLE_RATE * 2;

    header[0..4].copy_from_slice(b"RIFF");
    header[4..8].copy_from_slice(&(36 + data_len).to_le_bytes());
    header[8..12].copy_from_slice(b"WAVE");
    header[12..16].copy_from_slice(b"fmt ");
    header[16..20].copy_from_slice(&16u32.to_le_bytes());
    header[20..22].copy_from_slice(&1u16.to_le_bytes()); // PCM
    header[22..24].copy_from_slice(&1u16.to_le_bytes()); // mono
    header[24..28].copy_from_slice(&SAMPLE_RATE.to_le_bytes());
    header[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    header[32..34].copy_from_slice(&2u16.to_le_bytes()); // block align
    header[34..36].copy_from_slice(&16u16.to_le_bytes()); // bits per sample
    header[36..40].copy_from_slice(b"data");
    header[40..44].copy_from_slice(&data_len.to_le_bytes());
    header
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn writer(dir: &Path, name: &str) -> MonoWavWriter {
        MonoWavWriter::create(&dir.join(name)).unwrap()
    }

    #[test]
    fn audio_before_the_origin_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = writer(dir.path(), "a.wav");
        // A permission prompt is still open; this block predates the timeline.
        writer.append(&[0.5; 1600], 100.0).unwrap();
        assert_eq!(writer.frames, 0);
        assert_eq!(writer.seconds(), 0.0);
    }

    #[test]
    fn a_gap_is_padded_with_silence_so_later_audio_keeps_its_offset() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = writer(dir.path(), "b.wav");
        writer.set_origin(100.0);

        // First block arrives on time at t0.
        writer.append(&[0.5; 1600], 100.0).unwrap();
        assert_eq!(writer.frames, 1600);

        // Next block belongs a full second in — 0.9 s of it went missing.
        writer.append(&[0.5; 1600], 101.0).unwrap();
        assert_eq!(writer.frames, 16_000 + 1600, "silence must hold the timeline open");
        assert!((writer.drift_seconds() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn jitter_within_tolerance_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = writer(dir.path(), "c.wav");
        writer.set_origin(0.0);
        writer.append(&[0.1; 1600], 0.0).unwrap();
        // 50 ms early/late is ordinary callback jitter, under the 100 ms bar.
        writer.append(&[0.1; 1600], 0.05).unwrap();
        assert_eq!(writer.frames, 3200, "no padding or trimming expected");
        assert_eq!(writer.drift_seconds(), 0.0);
    }

    #[test]
    fn a_block_running_ahead_is_trimmed_rather_than_pushing_everything_later() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = writer(dir.path(), "d.wav");
        writer.set_origin(0.0);

        // A full second already written…
        writer.append(&[0.5; 16_000], 0.0).unwrap();
        // …but this block claims to start at 0.5 s, overlapping by 0.5 s.
        writer.append(&[0.5; 16_000], 0.5).unwrap();

        assert_eq!(writer.frames, 24_000, "the overlapping 8000 frames are dropped");
        assert!((writer.drift_seconds() + 0.5).abs() < 1e-6, "drift is signed");
    }

    #[test]
    fn a_block_shorter_than_the_overlap_is_dropped_entirely() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = writer(dir.path(), "e.wav");
        writer.set_origin(0.0);
        writer.append(&[0.5; 16_000], 0.0).unwrap();
        // Claims to start half a second back but only carries 100 frames.
        writer.append(&[0.5; 100], 0.5).unwrap();
        assert_eq!(writer.frames, 16_000);
    }

    #[test]
    fn close_patches_the_header_with_the_real_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.wav");
        let mut writer = MonoWavWriter::create(&path).unwrap();
        writer.set_origin(0.0);
        writer.append(&[0.25; 8000], 0.0).unwrap();
        let frames = writer.close().unwrap();

        assert_eq!(frames, 8000);
        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len() as u64, WAV_HEADER_LEN + 8000 * 2);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[36..40], b"data");
        assert_eq!(
            u32::from_le_bytes(bytes[40..44].try_into().unwrap()),
            8000 * 2,
            "data chunk length must match what was written"
        );
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            36 + 8000 * 2
        );
    }

    #[test]
    fn close_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = writer(dir.path(), "g.wav");
        writer.set_origin(0.0);
        writer.append(&[0.1; 160], 0.0).unwrap();
        assert_eq!(writer.close().unwrap(), 160);
        assert_eq!(writer.close().unwrap(), 160, "a second close must not corrupt");
    }

    #[test]
    fn peak_resets_each_time_it_is_read() {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = writer(dir.path(), "h.wav");
        writer.set_origin(0.0);
        writer.append(&[0.0, 0.7, -0.3], 0.0).unwrap();
        assert!((writer.take_peak() - 0.7).abs() < 1e-6);
        assert_eq!(writer.take_peak(), 0.0, "meters must fall when sound stops");
    }

    #[test]
    fn out_of_range_samples_clamp_instead_of_wrapping() {
        // A backend handing us 1.5 must not become a loud negative spike.
        assert_eq!(to_i16(1.5), i16::MAX);
        assert_eq!(to_i16(-1.5), -i16::MAX);
        assert_eq!(to_i16(0.0), 0);
    }

    #[test]
    fn the_two_tracks_stay_aligned_across_independent_jitter() {
        // The property the merged transcript actually depends on: two writers
        // sharing one t0 land the same wall-clock instant at the same frame,
        // however differently their callbacks arrived.
        let dir = tempfile::tempdir().unwrap();
        let mut mic = writer(dir.path(), "mic.wav");
        let mut system = writer(dir.path(), "sys.wav");
        mic.set_origin(1000.0);
        system.set_origin(1000.0);

        // Mic delivers tidy 0.1 s blocks.
        for step in 0..100 {
            mic.append(&[0.2; 1600], 1000.0 + step as f64 * 0.1).unwrap();
        }
        // System stalls for half a second in the middle, then catches up.
        for step in 0..50 {
            system.append(&[0.2; 1600], 1000.0 + step as f64 * 0.1).unwrap();
        }
        for step in 55..100 {
            system.append(&[0.2; 1600], 1000.0 + step as f64 * 0.1).unwrap();
        }

        // Both describe the same 10 seconds of wall clock.
        assert!(
            (mic.seconds() - system.seconds()).abs() < RESYNC_TOLERANCE_SEC * 2.0,
            "mic {} vs system {}",
            mic.seconds(),
            system.seconds()
        );
    }
}
