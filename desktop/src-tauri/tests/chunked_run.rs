//! End-to-end tests for the chunk loop's durability contract.
//!
//! These use a stub engine, so what is under test is the part that has to be
//! right regardless of which model runs: append -> fsync -> checkpoint ordering,
//! truncate-and-redo recovery, and cancellation keeping everything already done.
//!
//! Requires `ffmpeg` on PATH, which `decode_range` needs anyway.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use ai_meeting_lib::chunking::checkpoint::{checkpoint_path, load_checkpoint, source_fingerprint};
use ai_meeting_lib::chunking::{transcribe_chunked, ChunkObserver, ChunkOptions, TranscribeError};
use ai_meeting_lib::state::CancelFlag;
use ai_meeting_lib::transcribe::{Engine, SegmentSink, TranscriptSegment};
use ai_meeting_lib::SAMPLE_RATE;

const CHUNK_SEC: f64 = 5.0;

/// An engine that reports one segment per chunk, labelled with its offset, so
/// the transcript says exactly which chunks reached disk.
struct StubEngine {
    key: String,
    calls: AtomicUsize,
    /// Fail on the Nth call (1-based), standing in for a crash mid-run.
    fail_on: Option<usize>,
}

impl StubEngine {
    fn new(key: &str) -> Self {
        Self {
            key: key.to_string(),
            calls: AtomicUsize::new(0),
            fail_on: None,
        }
    }

    fn failing_on(key: &str, call: usize) -> Self {
        Self {
            key: key.to_string(),
            calls: AtomicUsize::new(0),
            fail_on: Some(call),
        }
    }
}

impl Engine for StubEngine {
    fn engine_key(&self) -> String {
        self.key.clone()
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn transcribe(
        &self,
        audio: &[f32],
        offset_sec: f64,
        on_segment: Option<&SegmentSink<'_>>,
    ) -> anyhow::Result<Vec<TranscriptSegment>> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_on == Some(call) {
            anyhow::bail!("stub engine failed on call {call}");
        }
        let end = offset_sec + audio.len() as f64 / SAMPLE_RATE as f64;
        let segment = TranscriptSegment::new(offset_sec, end, format!("chunk at {offset_sec:.2}"));
        if let Some(sink) = on_segment {
            sink(segment.clone());
        }
        Ok(vec![segment])
    }
}

#[derive(Default)]
struct CountingObserver {
    texts: std::sync::Mutex<Vec<String>>,
    previews: AtomicUsize,
}

impl ChunkObserver for CountingObserver {
    fn on_text(&self, text: &str) {
        self.texts.lock().unwrap().push(text.to_string());
    }
    fn on_segment(&self, _segment: &TranscriptSegment) {
        self.previews.fetch_add(1, Ordering::SeqCst);
    }
}

/// Write a mono 16 kHz 16-bit WAV of `seconds` of tone. Hand-rolled so the test
/// does not need a WAV crate just to produce something ffmpeg can read.
fn write_wav(path: &Path, seconds: f64) {
    let frames = (seconds * SAMPLE_RATE as f64) as u32;
    let data_len = frames * 2;
    let mut bytes = Vec::with_capacity(44 + data_len as usize);

    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    bytes.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes()); // byte rate
    bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
    bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());

    for frame in 0..frames {
        let phase = frame as f64 / SAMPLE_RATE as f64 * 220.0 * std::f64::consts::TAU;
        let sample = (phase.sin() * 12_000.0) as i16;
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

fn fixture(seconds: f64) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let audio = dir.path().join("meeting.wav");
    write_wav(&audio, seconds);
    (dir, audio)
}

fn options(cancel: Option<CancelFlag>) -> ChunkOptions {
    ChunkOptions {
        chunk_sec: CHUNK_SEC,
        output_path: None,
        cancel,
    }
}

#[test]
fn a_completed_run_writes_every_chunk_and_clears_the_checkpoint() {
    let (_dir, audio) = fixture(12.0);
    let engine = StubEngine::new("stub:vi");
    let observer = CountingObserver::default();

    let text = transcribe_chunked(&audio, &engine, &observer, &options(None)).unwrap();

    assert!(text.contains("[00:00:00] chunk at 0.00"), "{text}");
    assert!(text.contains("chunk at 5.05"), "{text}");
    assert!(
        text.contains("----- Complete:"),
        "a finished transcript is marked whole: {text}"
    );
    assert!(
        !checkpoint_path(&audio).exists(),
        "a finished file must not leave a checkpoint behind"
    );
    assert!(
        observer.previews.load(Ordering::SeqCst) >= 2,
        "streaming engines must report previews"
    );
}

#[test]
fn an_interrupted_run_resumes_from_the_last_checkpointed_chunk() {
    let (_dir, audio) = fixture(12.0);

    // Die on the third chunk, after two are safely on disk.
    let failing = StubEngine::failing_on("stub:vi", 3);
    let observer = CountingObserver::default();
    let error = transcribe_chunked(&audio, &failing, &observer, &options(None)).unwrap_err();
    assert!(matches!(error, TranscribeError::Other(_)));

    let fingerprint = source_fingerprint(&audio, "stub:vi", CHUNK_SEC).unwrap();
    let state = load_checkpoint(&audio, &fingerprint).expect("progress must be checkpointed");
    assert_eq!(state.chunks_done, 2);

    // The transcript on disk is a complete prefix — readable, no partial line.
    let partial = fs::read_to_string(audio.with_file_name(&state.transcript_name)).unwrap();
    assert_eq!(partial.lines().count(), 2, "{partial}");
    assert_eq!(partial.len() as u64, state.text_bytes);

    // Re-running continues rather than starting over, and appends to the *same*
    // transcript the interrupted run created.
    let resumed_engine = StubEngine::new("stub:vi");
    let resumed_observer = CountingObserver::default();
    let text =
        transcribe_chunked(&audio, &resumed_engine, &resumed_observer, &options(None)).unwrap();

    assert!(text.starts_with("[00:00:00] chunk at 0.00"), "{text}");
    assert!(text.contains("----- Complete:"));
    assert!(
        text.contains("resumed from"),
        "the footer must admit it did not do the whole file: {text}"
    );
    assert_eq!(
        resumed_engine.calls.load(Ordering::SeqCst),
        1,
        "only the one remaining chunk should be transcribed again"
    );
}

#[test]
fn a_tail_past_the_checkpoint_is_truncated_and_that_chunk_redone() {
    let (_dir, audio) = fixture(12.0);

    let failing = StubEngine::failing_on("stub:vi", 3);
    let _ = transcribe_chunked(&audio, &failing, &CountingObserver::default(), &options(None));

    let fingerprint = source_fingerprint(&audio, "stub:vi", CHUNK_SEC).unwrap();
    let state = load_checkpoint(&audio, &fingerprint).unwrap();
    let transcript = audio.with_file_name(&state.transcript_name);

    // Simulate a crash *between* the append and the checkpoint write: the file
    // has a chunk the checkpoint does not know about, and it is half-written.
    let mut debris = fs::read_to_string(&transcript).unwrap();
    debris.push_str("[00:00:10] half-writ");
    fs::write(&transcript, &debris).unwrap();

    let engine = StubEngine::new("stub:vi");
    let text = transcribe_chunked(&audio, &engine, &CountingObserver::default(), &options(None))
        .unwrap();

    assert!(
        !text.contains("half-writ"),
        "the debris past text_bytes must be truncated away: {text}"
    );
    // Every line is a whole line, and no chunk appears twice.
    let chunk_lines: Vec<&str> = text.lines().filter(|l| l.contains("chunk at")).collect();
    let mut unique = chunk_lines.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(chunk_lines.len(), unique.len(), "duplicated chunk: {text}");
}

#[test]
fn cancelling_keeps_everything_already_written() {
    let (_dir, audio) = fixture(12.0);
    let cancel = CancelFlag::new();
    // Trip it immediately: the first between-chunks check should stop the run.
    cancel.cancel();

    let engine = StubEngine::new("stub:vi");
    let error = transcribe_chunked(
        &audio,
        &engine,
        &CountingObserver::default(),
        &options(Some(cancel)),
    )
    .unwrap_err();

    assert!(matches!(error, TranscribeError::Cancelled));
    assert_eq!(
        engine.calls.load(Ordering::SeqCst),
        0,
        "cancellation is checked before the chunk is decoded"
    );
}

#[test]
fn switching_engine_restarts_instead_of_resuming() {
    let (_dir, audio) = fixture(12.0);

    let failing = StubEngine::failing_on("stub:vi", 3);
    let _ = transcribe_chunked(&audio, &failing, &CountingObserver::default(), &options(None));
    assert!(checkpoint_path(&audio).exists());

    // A different engine key must not resume into text produced by the old one.
    let other = StubEngine::new("stub:en");
    let text =
        transcribe_chunked(&audio, &other, &CountingObserver::default(), &options(None)).unwrap();

    assert!(
        !text.contains("resumed from"),
        "a different engine must start the file over: {text}"
    );
    assert_eq!(
        other.calls.load(Ordering::SeqCst),
        3,
        "the whole file should be redone"
    );
}
