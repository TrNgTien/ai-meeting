//! Spawns and owns the `sidecar` child process: a PyInstaller-built,
//! self-contained executable (see `desktop/scripts/build-sidecar.sh`) wrapping
//! `sidecar.py`, the persistent Python subprocess that does the actual
//! transcription/model-management work by calling the existing, unchanged
//! `transcriber.py`/`chunking.py`/`mlx_engine.py`/`phowhisper.py` modules.
//! Bundled as a Tauri resource rather than shelling out to the repo's `.venv`,
//! so the built `.app` doesn't depend on the source checkout.
//!
//! One process per app session: `sidecar.py`'s own single-worker-thread
//! assumption (it taps `transcriber.stream_segments()`'s stdout redirection
//! trick, which is only safe with one transcription in flight) means this
//! side must never spawn a second one.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Arc;

use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde_json::json;
use tauri::{AppHandle, Emitter, Manager};

/// The event every parsed sidecar stdout line (and the synthetic
/// disconnect notice below) is forwarded to the frontend as.
pub const SIDECAR_EVENT: &str = "sidecar-event";

/// Owns the child's stdin so Tauri commands can write JSON lines to it.
/// The stdout side is not stored here — Task 2's reader thread owns that
/// pipe for the process's lifetime.
pub struct SidecarState {
    stdin: Mutex<ChildStdin>,
    /// Kept alive only so the child is killed if the app exits uncleanly;
    /// nothing reads from it directly (the reader thread already owns
    /// stdout, and this app doesn't need to `wait()` on the child).
    _child: Arc<Mutex<Child>>,
}

impl SidecarState {
    /// Serialise `msg` to one JSON line and write it to the sidecar's
    /// stdin. `sidecar.py`'s command loop is one JSON object per line —
    /// see `main()` in `sidecar.py`.
    pub fn send(&self, msg: serde_json::Value) -> Result<(), String> {
        let mut line = msg.to_string();
        line.push('\n');
        let mut stdin = self.stdin.lock();
        stdin.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
        stdin.flush().map_err(|e| e.to_string())
    }
}

/// The PyInstaller onedir bundle at `resources/sidecar/` (see
/// `desktop/scripts/build-sidecar.sh` and `tauri.conf.json`'s
/// `bundle.resources`) is a self-contained executable — no system Python or
/// `.venv` required. `tauri dev` stages configured resources next to the dev
/// binary the same way `tauri build` does for the bundled `.app`, so this one
/// path resolves correctly in both cases.
fn sidecar_binary_path(app: &AppHandle) -> Result<PathBuf> {
    let resource_dir = app
        .path()
        .resource_dir()
        .context("could not resolve the app's resource directory")?;
    let binary = resource_dir.join("sidecar").join("sidecar");
    anyhow::ensure!(
        binary.exists(),
        "sidecar binary not found at {} — run `make desktop-build-sidecar` first",
        binary.display()
    );
    Ok(binary)
}

/// Spawn the bundled `sidecar` executable and start the stdout-reader thread
/// that forwards each line to the frontend as a [`SIDECAR_EVENT`].
pub fn spawn(app: &AppHandle) -> Result<SidecarState> {
    let sidecar_bin = sidecar_binary_path(app)?;
    let sidecar_dir = sidecar_bin
        .parent()
        .context("sidecar binary has no parent directory")?;

    let mut child = Command::new(&sidecar_bin)
        .current_dir(sidecar_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("cannot spawn {}", sidecar_bin.display()))?;

    let stdin = child.stdin.take().context("sidecar child has no stdin")?;
    let stdout = child.stdout.take().context("sidecar child has no stdout")?;
    let stderr = child.stderr.take().context("sidecar child has no stderr")?;

    spawn_stdout_reader(app.clone(), stdout);
    spawn_stderr_logger(stderr);

    Ok(SidecarState {
        stdin: Mutex::new(stdin),
        _child: Arc::new(Mutex::new(child)),
    })
}

/// Parses each stdout line as JSON and re-emits it verbatim as a
/// [`SIDECAR_EVENT`] — the frontend destructures by the line's own
/// `"event"` field rather than this bridge maintaining a typed Rust struct
/// per event, so new `sidecar.py` events don't require a Rust-side change.
fn spawn_stdout_reader(app: AppHandle, stdout: impl std::io::Read + Send + 'static) {
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<serde_json::Value>(&line) {
                Ok(value) => {
                    let _ = app.emit(SIDECAR_EVENT, value);
                }
                Err(err) => {
                    eprintln!("sidecar: bad JSON line ({err}): {line}");
                }
            }
        }
        // stdout closed: the sidecar process exited (crash or otherwise).
        // Every in-flight command's result depends on a matching event
        // arriving, so the frontend needs a distinct signal to stop
        // waiting rather than hang. Underscore-prefixed so it reads as
        // bridge-internal, not a real sidecar.py event.
        let _ = app.emit(SIDECAR_EVENT, json!({"event": "_sidecar_exited"}));
    });
}

/// `sidecar.py` never writes to stderr in normal operation — any line here
/// is a real crash/traceback, worth surfacing in the dev console rather
/// than silently dropping.
fn spawn_stderr_logger(stderr: impl std::io::Read + Send + 'static) {
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            eprintln!("sidecar stderr: {line}");
        }
    });
}
