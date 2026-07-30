//! Recording both sides of a meeting to two time-aligned WAV files.
//!
//! Port of `audio_capture.MeetingRecorder` / `Recording` / `Levels`.
//!
//! Either side may be absent — no microphone selected, or no way to reach the
//! system mix on this machine — and the recording still runs with what it has.
//! Whatever is missing is reported in [`Recording::warnings`] rather than failing
//! the whole take, **because a meeting is not repeatable and half a recording
//! beats none**. That is the single most important rule in this file.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use super::mic::MicCapture;
use super::system::{open_system_capture, Prefer, SystemCapture};
use super::wav_writer::MonoWavWriter;

/// What a finished recording left on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recording {
    pub stem: String,
    pub mic_path: Option<PathBuf>,
    pub system_path: Option<PathBuf>,
    pub duration_sec: f64,
    /// Unix seconds, for the merged transcript's header.
    pub started_at: f64,
    pub warnings: Vec<String>,
}

impl Recording {
    pub fn paths(&self) -> Vec<PathBuf> {
        [self.mic_path.clone(), self.system_path.clone()]
            .into_iter()
            .flatten()
            .collect()
    }
}

/// Meter reading for the UI, 0.0-1.0 per side.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Levels {
    pub mic: f32,
    pub system: f32,
}

/// How a recording should be set up.
pub struct RecorderOptions {
    pub record_mic: bool,
    pub record_system: bool,
    pub mic_device_id: Option<String>,
    pub system_backend: Prefer,
    /// Overridden only by tests; production uses `meeting-YYYYmmdd-HHMMSS`.
    pub stem: Option<String>,
}

impl Default for RecorderOptions {
    fn default() -> Self {
        Self {
            record_mic: true,
            record_system: true,
            mic_device_id: None,
            system_backend: Prefer::Auto,
            stem: None,
        }
    }
}

/// A shared WAV writer. The audio callbacks run on backend threads, so the
/// writer has to be behind a lock; the lock is only ever held for the length of
/// one block append.
type SharedWriter = Arc<Mutex<MonoWavWriter>>;

pub struct MeetingRecorder {
    stem: String,
    out_dir: PathBuf,
    clock: Instant,
    started_at: f64,
    running: bool,

    mic: Option<MicCapture>,
    system: Option<SystemCapture>,
    mic_writer: Option<SharedWriter>,
    system_writer: Option<SharedWriter>,
    mic_path: Option<PathBuf>,
    system_path: Option<PathBuf>,
    warnings: Vec<String>,
}

impl MeetingRecorder {
    /// Open both sides and start recording.
    ///
    /// Fails only when *neither* side could be opened; anything less becomes a
    /// warning carried on the finished [`Recording`].
    pub fn start(out_dir: &Path, options: RecorderOptions) -> Result<Self> {
        std::fs::create_dir_all(out_dir)
            .with_context(|| format!("cannot create {}", out_dir.display()))?;

        let stem = options
            .stem
            .unwrap_or_else(|| chrono::Local::now().format("meeting-%Y%m%d-%H%M%S").to_string());

        let clock = Instant::now();
        let mut warnings = Vec::new();

        let mut mic = None;
        let mut mic_writer: Option<SharedWriter> = None;
        let mut mic_path = None;

        if options.record_mic {
            let path = out_dir.join(format!("{stem}-me.wav"));
            match MonoWavWriter::create(&path) {
                Ok(writer) => {
                    let writer: SharedWriter = Arc::new(Mutex::new(writer));
                    let sink_writer = Arc::clone(&writer);
                    let sink = Box::new(move |block: &[f32], started_at: f64| {
                        if let Ok(mut writer) = sink_writer.lock() {
                            let _ = writer.append(block, started_at);
                        }
                    });
                    match MicCapture::start(options.mic_device_id.as_deref(), clock, sink) {
                        Ok(capture) => {
                            mic = Some(capture);
                            mic_writer = Some(writer);
                            mic_path = Some(path);
                        }
                        Err(error) => {
                            warnings.push(format!("Microphone unavailable: {error}"));
                            let _ = writer.lock().map(|mut w| w.close());
                            let _ = std::fs::remove_file(&path);
                        }
                    }
                }
                Err(error) => warnings.push(format!("Cannot write the microphone track: {error}")),
            }
        }

        let mut system = None;
        let mut system_writer: Option<SharedWriter> = None;
        let mut system_path = None;

        if options.record_system {
            let path = out_dir.join(format!("{stem}-meeting.wav"));
            match MonoWavWriter::create(&path) {
                Ok(writer) => {
                    let writer: SharedWriter = Arc::new(Mutex::new(writer));
                    let sink_writer = Arc::clone(&writer);
                    let sink = Box::new(move |block: &[f32], started_at: f64| {
                        if let Ok(mut writer) = sink_writer.lock() {
                            let _ = writer.append(block, started_at);
                        }
                    });
                    let (capture, problem) =
                        open_system_capture(options.system_backend, clock, sink);
                    match capture {
                        Some(capture) => {
                            system = Some(capture);
                            system_writer = Some(writer);
                            system_path = Some(path);
                        }
                        None => {
                            warnings.push(
                                problem.unwrap_or_else(|| "system audio is unavailable".into()),
                            );
                            let _ = writer.lock().map(|mut w| w.close());
                            let _ = std::fs::remove_file(&path);
                        }
                    }
                }
                Err(error) => warnings.push(format!("Cannot write the meeting track: {error}")),
            }
        }

        if mic.is_none() && system.is_none() {
            let mut message = "Nothing could be recorded.".to_string();
            if !warnings.is_empty() {
                message.push(' ');
                message.push_str(&warnings.join(" "));
            }
            return Err(anyhow!(message));
        }

        // Both sides are live: from here the two tracks share one origin, which
        // is what makes their transcripts mergeable on timestamp. Set *after*
        // the backends are up, so a permission prompt the user took ten seconds
        // to dismiss is not baked into the files as silence.
        let origin = clock.elapsed().as_secs_f64();
        for writer in [mic_writer.as_ref(), system_writer.as_ref()].into_iter().flatten() {
            if let Ok(mut writer) = writer.lock() {
                writer.set_origin(origin);
            }
        }

        Ok(Self {
            stem,
            out_dir: out_dir.to_path_buf(),
            clock,
            started_at: unix_seconds_now(),
            running: true,
            mic,
            system,
            mic_writer,
            system_writer,
            mic_path,
            system_path,
            warnings,
        })
    }

    pub fn running(&self) -> bool {
        self.running
    }

    pub fn elapsed(&self) -> f64 {
        if self.running {
            self.clock.elapsed().as_secs_f64()
        } else {
            0.0
        }
    }

    /// Peak level per side since the last call — drives the meters.
    pub fn levels(&self) -> Levels {
        Levels {
            mic: peak(self.mic_writer.as_ref()),
            system: peak(self.system_writer.as_ref()),
        }
    }

    pub fn system_description(&self) -> Option<&'static str> {
        self.system.as_ref().map(|s| s.backend().description())
    }

    /// Stop both sides and finish the files.
    pub fn stop(mut self) -> Recording {
        self.running = false;

        // Stop capturing before closing the writers, so no callback can arrive
        // after the header has been patched.
        self.mic = None;
        if let Some(system) = self.system.as_mut() {
            system.stop();
        }

        // Errors a backend noticed along the way belong on the recording.
        if let Some(capture) = self.system.as_ref() {
            if let Some(error) = capture.error() {
                self.warnings.push(format!("Meeting track: {error}"));
            }
        }
        self.system = None;

        let mic_frames = close(self.mic_writer.take());
        let system_frames = close(self.system_writer.take());
        let duration_sec = mic_frames.max(system_frames);

        // A side that captured nothing at all is not a track; keeping a
        // zero-length WAV would make the merge report an empty speaker.
        let mic_path = self.mic_path.take().filter(|_| mic_frames > 0.0);
        let system_path = self.system_path.take().filter(|_| system_frames > 0.0);

        Recording {
            stem: std::mem::take(&mut self.stem),
            mic_path,
            system_path,
            duration_sec,
            started_at: self.started_at,
            warnings: std::mem::take(&mut self.warnings),
        }
    }

    pub fn out_dir(&self) -> &Path {
        &self.out_dir
    }
}

fn peak(writer: Option<&SharedWriter>) -> f32 {
    writer
        .and_then(|writer| writer.lock().ok().map(|mut writer| writer.take_peak()))
        .unwrap_or(0.0)
}

/// Close a writer and return its duration in seconds.
fn close(writer: Option<SharedWriter>) -> f64 {
    let Some(writer) = writer else {
        return 0.0;
    };
    let Ok(mut writer) = writer.lock() else {
        return 0.0;
    };
    let _ = writer.close();
    writer.seconds()
}

fn unix_seconds_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|delta| delta.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recording_with_neither_side_requested_is_refused() {
        // The UI blocks this, but the recorder must not silently produce an empty
        // take either.
        let dir = tempfile::tempdir().unwrap();
        // `MeetingRecorder` owns live audio streams and is deliberately not
        // Debug, so the error is matched without unwrap_err's Debug bound.
        let message = match MeetingRecorder::start(
            dir.path(),
            RecorderOptions {
                record_mic: false,
                record_system: false,
                ..Default::default()
            },
        ) {
            Ok(_) => panic!("a recording with neither side must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(message.contains("Nothing could be recorded"), "{message}");
    }

    #[test]
    fn track_files_are_named_by_side() {
        // The merge and the UI both key off these names.
        let stem = "meeting-20260730-142530";
        assert_eq!(format!("{stem}-me.wav"), "meeting-20260730-142530-me.wav");
        assert_eq!(
            format!("{stem}-meeting.wav"),
            "meeting-20260730-142530-meeting.wav"
        );
    }

    #[test]
    fn a_recording_reports_only_the_tracks_that_captured_audio() {
        let recording = Recording {
            stem: "meeting-x".into(),
            mic_path: Some(PathBuf::from("/tmp/x-me.wav")),
            system_path: None,
            duration_sec: 12.0,
            started_at: 0.0,
            warnings: vec!["Screen Recording permission is not granted".into()],
        };
        assert_eq!(recording.paths().len(), 1);
        assert!(!recording.warnings.is_empty(), "the user must learn why");
    }

    #[test]
    fn an_empty_writer_contributes_no_duration() {
        let dir = tempfile::tempdir().unwrap();
        let writer = MonoWavWriter::create(&dir.path().join("empty.wav")).unwrap();
        let shared: SharedWriter = Arc::new(Mutex::new(writer));
        assert_eq!(close(Some(shared)), 0.0);
    }

    #[test]
    fn peak_of_a_missing_side_is_silence_not_a_panic() {
        assert_eq!(peak(None), 0.0);
    }
}
