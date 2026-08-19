//! Headless transcription — what `make transcribe` runs.
//!
//! A real binary rather than an example, because it is the CLI: `transcriber.py`
//! is going away and this takes over the job of transcribing a file without
//! opening a window. It doubles as the A/B harness — run it and the Python app
//! on the same file with the same model, then diff the transcripts.
//!
//!     cargo run --release --bin transcribe -- <audio> [model] [lang] [chunk_sec]
//!
//! `lang` is one of `vi+en` (default), `en`, `auto`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use transcriber_lib::chunking::{
    transcribe_chunked, ChunkObserver, ChunkOptions, DEFAULT_CHUNK_SECONDS,
};
use transcriber_lib::state::LanguageMode;
use transcriber_lib::transcribe::models::{ensure_model_downloaded, is_model_downloaded};
use transcriber_lib::transcribe::whisper_cpp::WhisperCppEngine;
use transcriber_lib::transcribe::{TranscriptSegment, DEFAULT_MODEL};

/// Prints progress the way the Python app's status bar reads.
struct Printing {
    previews: AtomicUsize,
}

impl ChunkObserver for Printing {
    fn on_status(&self, message: &str) {
        println!("{message}");
    }

    fn on_text(&self, text: &str) {
        print!("{text}");
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }

    fn on_segment(&self, _segment: &TranscriptSegment) {
        // Counted rather than printed: the preview would interleave with the
        // chunk text and make the output impossible to diff.
        self.previews.fetch_add(1, Ordering::Relaxed);
    }

    fn on_progress(&self, chunks_done: usize, done_sec: f64, total_sec: Option<f64>) {
        let total = total_sec
            .map(transcriber_lib::transcribe::format_timestamp)
            .unwrap_or_else(|| "??:??:??".into());
        eprintln!(
            "  [chunk {chunks_done}] {} / {total} transcribed (saved)",
            transcriber_lib::transcribe::format_timestamp(done_sec)
        );
    }
}

fn parse_language(value: &str) -> LanguageMode {
    LanguageMode::parse(value)
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let source = PathBuf::from(
        args.next()
            .ok_or_else(|| anyhow::anyhow!("usage: transcribe <audio> [model] [lang] [chunk_sec]"))?,
    );
    let model = args.next().unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let language = parse_language(&args.next().unwrap_or_else(|| "vi+en".into()));
    let chunk_sec: f64 = args
        .next()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(DEFAULT_CHUNK_SECONDS);

    if !is_model_downloaded(&model) {
        eprintln!("Downloading '{model}' (one-time)…");
    }
    let report = |name: &str, done: u64, total: u64| {
        if total > 0 {
            eprint!(
                "\r  {name}: {:.0}/{:.0} MB ({:.0}%)",
                done as f64 / 1e6,
                total as f64 / 1e6,
                done as f64 / total as f64 * 100.0
            );
        } else {
            eprint!("\r  {name}: {:.0} MB", done as f64 / 1e6);
        }
    };
    let model_path = ensure_model_downloaded(&model, Some(&report), None)?;
    eprintln!();

    eprintln!(
        "Loading '{model}' ({:?} -> language {:?})…",
        language,
        language.whisper_language()
    );
    let engine = WhisperCppEngine::load(&model_path, &model, language.whisper_language())?;
    eprintln!("engine_key = {}", engine.engine_key());

    let observer = Printing {
        previews: AtomicUsize::new(0),
    };
    let options = ChunkOptions {
        chunk_sec,
        ..Default::default()
    };

    let started = std::time::Instant::now();
    transcribe_chunked(&source, &engine, &observer, &options)?;
    eprintln!(
        "done in {:.1}s, {} preview segments streamed",
        started.elapsed().as_secs_f64(),
        observer.previews.load(Ordering::Relaxed)
    );
    Ok(())
}

// `Engine` has to be in scope for `engine.engine_key()` above.
use transcriber_lib::transcribe::Engine;
