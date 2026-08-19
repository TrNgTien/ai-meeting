//! The thread that owns a live recording, and the commands' way in to it.
//!
//! Port of the recording half of `app.TranscriberApp` — `_start_recording`,
//! `_stop_recording` and the meter tick — with one structural change the
//! platform forces.
//!
//! A [`MeetingRecorder`] owns a CoreAudio input stream and, on macOS 13+, a
//! ScreenCaptureKit stream. Neither is `Send`, so the recorder cannot live in
//! Tauri's shared state the way [`crate::engine::EngineHost`]'s job slot does.
//! Instead one long-lived thread owns it and everything else talks to that
//! thread over a channel: the recorder is created, polled and dropped without
//! ever crossing a thread boundary.
//!
//! That indirection buys something beyond satisfying the compiler. Opening the
//! system side can block for as long as it takes the user to dismiss a Screen
//! Recording permission dialog, and closing it waits for the backend to flush.
//! Doing either on the thread serving Tauri commands would freeze the window at
//! exactly the moment the user is being asked to click something, so `start` and
//! `stop` report through callbacks instead of returning a value.

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Duration;

use parking_lot::Mutex;
use serde::Serialize;

use crate::audio::recorder::{MeetingRecorder, Recording, RecorderOptions};
use crate::audio::system::Prefer;

/// How long a meter poll waits for the recording thread.
///
/// The meters tick ten times a second and are decoration; if the thread is busy
/// opening a backend, the honest answer is "no reading yet" and the next tick
/// will have one. Blocking the UI for a smoother bar would be the wrong trade.
const LEVELS_TIMEOUT: Duration = Duration::from_millis(50);

/// What the UI needs the moment recording actually starts.
#[derive(Debug, Clone, Serialize)]
pub struct Started {
    pub stem: String,
    /// Which backend the meeting side ended up on, for the status line.
    pub system_description: Option<String>,
    /// Non-fatal problems: a side that could not be opened, most often the
    /// meeting one. Recording continues with whatever did open.
    pub warnings: Vec<String>,
}

/// One meter reading, polled by the frontend while recording.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct LiveLevels {
    pub mic: f32,
    pub system: f32,
    pub elapsed_sec: f64,
    /// False once the recording has stopped, so a late poll cannot leave the
    /// timer frozen at its last value.
    pub running: bool,
}

type StartDone = Box<dyn FnOnce(Result<Started, String>) + Send>;
type StopDone = Box<dyn FnOnce(Option<Recording>) + Send>;

enum Request {
    Start {
        out_dir: PathBuf,
        options: RecorderOptions,
        done: StartDone,
    },
    Levels {
        reply: Sender<LiveLevels>,
    },
    Stop {
        done: StopDone,
    },
}

/// Shared state for the recording thread. Owned by Tauri; cloneable handles are
/// not needed because every method sends and returns.
#[derive(Default)]
pub struct RecordingHost {
    /// `None` until the first request spawns the thread — an app that only ever
    /// transcribes imported files never touches an audio device.
    tx: Mutex<Option<Sender<Request>>>,
}

impl RecordingHost {
    fn sender(&self) -> Sender<Request> {
        let mut slot = self.tx.lock();
        if let Some(tx) = slot.as_ref() {
            return tx.clone();
        }
        let (tx, rx) = channel();
        std::thread::Builder::new()
            .name("recording".to_string())
            .spawn(move || serve(rx))
            .expect("cannot spawn the recording thread");
        *slot = Some(tx.clone());
        tx
    }

    /// Open both sides and start writing. `done` fires on the recording thread
    /// once the backends are up, or with the reason they are not.
    pub fn start(&self, out_dir: PathBuf, options: RecorderOptions, done: StartDone) {
        let _ = self.sender().send(Request::Start {
            out_dir,
            options,
            done,
        });
    }

    /// Stop and finish the files. `done` receives `None` if nothing was running.
    pub fn stop(&self, done: StopDone) {
        let _ = self.sender().send(Request::Stop { done });
    }

    pub fn levels(&self) -> LiveLevels {
        let (reply, rx) = channel();
        if self.sender().send(Request::Levels { reply }).is_err() {
            return LiveLevels::default();
        }
        rx.recv_timeout(LEVELS_TIMEOUT).unwrap_or_default()
    }
}

fn serve(rx: Receiver<Request>) {
    let mut recorder: Option<MeetingRecorder> = None;

    while let Ok(request) = rx.recv() {
        match request {
            Request::Start {
                out_dir,
                options,
                done,
            } => {
                if recorder.is_some() {
                    done(Err("a recording is already running".to_string()));
                    continue;
                }
                match MeetingRecorder::start(&out_dir, options) {
                    Ok(started) => {
                        let info = Started {
                            stem: started.stem().to_string(),
                            system_description: started.system_description().map(str::to_string),
                            warnings: started.warnings().to_vec(),
                        };
                        recorder = Some(started);
                        done(Ok(info));
                    }
                    // `{:#}` so the causes anyhow collected — which backend
                    // failed and why — reach the user, not just the outermost
                    // "could not start recording".
                    Err(error) => done(Err(format!("{error:#}"))),
                }
            }
            Request::Levels { reply } => {
                let snapshot = recorder
                    .as_ref()
                    .map(|recorder| {
                        let levels = recorder.levels();
                        LiveLevels {
                            mic: levels.mic,
                            system: levels.system,
                            elapsed_sec: recorder.elapsed(),
                            running: true,
                        }
                    })
                    .unwrap_or_default();
                let _ = reply.send(snapshot);
            }
            Request::Stop { done } => done(recorder.take().map(MeetingRecorder::stop)),
        }
    }
}

/// Where recordings and their transcripts go.
///
/// The Python app wrote to a `recordings/` folder beside its own source, which
/// only worked because it ran from a checkout. A bundled `.app` cannot write
/// next to itself, and a user should be able to find a meeting they recorded
/// without being told a path — so it lands somewhere they already look.
pub fn recordings_dir() -> PathBuf {
    dirs::document_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("Transcriber")
        .join("recordings")
}

/// Parse the system-audio backend the frontend asks for. Anything unrecognised
/// is `Auto`, which tries ScreenCaptureKit and falls back to a loopback device.
pub fn parse_backend(value: &str) -> Prefer {
    match value {
        "screencapturekit" => Prefer::ScreenCaptureKit,
        "loopback" => Prefer::Loopback,
        _ => Prefer::Auto,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_backends_fall_back_to_auto() {
        assert_eq!(parse_backend("loopback"), Prefer::Loopback);
        assert_eq!(parse_backend("screencapturekit"), Prefer::ScreenCaptureKit);
        assert_eq!(parse_backend("auto"), Prefer::Auto);
        assert_eq!(parse_backend(""), Prefer::Auto);
    }

    #[test]
    fn recordings_live_somewhere_the_user_can_find_them() {
        let dir = recordings_dir();
        assert!(dir.is_absolute() || dir.starts_with("."));
        assert!(dir.ends_with("Transcriber/recordings"));
    }

    #[test]
    fn stopping_when_nothing_runs_reports_no_recording() {
        // The UI blocks this, but a stray Stop must not panic the thread that
        // owns the audio devices — losing it would take live recording with it
        // until the app restarts.
        let host = RecordingHost::default();
        let (tx, rx) = channel();
        host.stop(Box::new(move |recording| {
            let _ = tx.send(recording.is_none());
        }));
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)), Ok(true));
    }

    #[test]
    fn levels_are_zero_and_not_running_before_a_recording_starts() {
        let host = RecordingHost::default();
        let levels = host.levels();
        assert!(!levels.running);
        assert_eq!(levels.elapsed_sec, 0.0);
    }
}
