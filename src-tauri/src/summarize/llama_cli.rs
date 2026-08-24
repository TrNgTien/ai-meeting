//! The one real [`Summarizer`]: a bundled llama-cli running a GGUF.
//!
//! The app bundles llama.cpp's `llama-cli` (see `scripts/build-llama.sh`) the
//! way it bundles ffmpeg, and shells out to it — the same sidecar pattern as the
//! decoders. Keeping it a separate process means the app binary carries no
//! second cmake-built ggml (whisper-rs already statically links one), and it
//! needs no `loaded` cache slot because there is no in-process state to keep
//! warm: the sidecar is spawned per summary, holds the weights only while the
//! summary runs, and exits.

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use anyhow::Context;

use super::{SummarizeError, Summarizer, TokenSink};
use crate::state::CancelFlag;

/// How often the watcher thread checks for a Stop while llama-cli is running.
/// Short enough that pressing Stop feels immediate, long enough to be free.
const CANCEL_POLL: Duration = Duration::from_millis(100);

/// Cap on generated tokens per completion. Generous for a summary, small enough
/// that a runaway generation cannot spin forever.
const MAX_TOKENS: u32 = 2048;

/// Build the llama-cli argument vector. Pure so it is unit-testable without
/// spawning a process: every argument is a side effect only when executed.
///
/// `--jinja` applies the chat template stored in the GGUF metadata, which is
/// what makes "bring your own model" actually work — swapping in any other
/// instruct GGUF needs no code change. `-no-cnv` requests a single completion
/// (not an interactive chat loop) and `-st` (single turn) makes it exit when
/// done. The (large) transcript goes in `prompt_file`, never on argv — a real
/// meeting's text blows past `E2BIG`.
pub fn build_args(
    model_path: &Path,
    system: &str,
    prompt_file: &Path,
    context: u32,
    threads: i32,
) -> Vec<String> {
    vec![
        "-m".into(),
        model_path.display().to_string(),
        "-sys".into(),
        system.to_string(),
        "-f".into(),
        prompt_file.display().to_string(),
        "--jinja".into(),
        "-no-cnv".into(),
        "-ngl".into(),
        "99".into(),
        "-c".into(),
        context.to_string(),
        "-n".into(),
        MAX_TOKENS.to_string(),
        "--temp".into(),
        "0.3".into(),
        "--top-p".into(),
        "0.9".into(),
        "-t".into(),
        threads.to_string(),
        "--no-warmup".into(),
        "--simple-io".into(),
        "-st".into(),
    ]
}

/// A prompt on disk, removed when it goes out of scope.
///
/// The transcript is >100 KB on any real meeting, so it must reach llama-cli by
/// file, not argv. Writing to the system temp dir and deleting on drop keeps a
/// cancelled or crashed run from leaking prompt files.
struct PromptFile(PathBuf);

impl PromptFile {
    fn write(text: &str) -> Result<Self, SummarizeError> {
        let path = std::env::temp_dir().join(format!(
            "transcriber-prompt-{}.txt",
            std::process::id()
        ));
        std::fs::write(&path, text)
            .with_context(|| format!("cannot write prompt file {}", path.display()))?;
        Ok(PromptFile(path))
    }
}

impl Drop for PromptFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub struct LlamaCliSummarizer {
    pub binary: PathBuf,
    pub model_path: PathBuf,
    pub model_id: String,
    pub context: u32,
    pub threads: i32,
}

impl LlamaCliSummarizer {
    /// Resolve the bundled llama-cli the way the decoders resolve ffmpeg.
    pub fn resolve_binary() -> PathBuf {
        crate::chunking::decode::sidecar("llama-cli", "TRANSCRIBER_LLAMA_DIR")
    }

    pub fn new(model_path: PathBuf, model_id: String, context: u32) -> Self {
        Self {
            binary: Self::resolve_binary(),
            model_path,
            model_id,
            context,
            threads: crate::transcribe::params::default_threads(),
        }
    }
}

impl Summarizer for LlamaCliSummarizer {
    fn key(&self) -> String {
        self.model_id.clone()
    }

    fn context(&self) -> u32 {
        self.context
    }

    fn generate(
        &self,
        system: &str,
        user: &str,
        on_token: Option<&TokenSink<'_>>,
        cancel: &CancelFlag,
    ) -> Result<String, SummarizeError> {
        let prompt = PromptFile::write(user)?;

        let args = build_args(&self.model_path, system, &prompt.0, self.context, self.threads);
        let mut child = Command::new(&self.binary)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("cannot run {}", self.binary.display()))?;

        // llama-cli writes *everything* to stdout in v0.2.0 — its banner, the
        // echoed prompt, the completion, then a timing line and "Exiting...".
        // stderr is where model-load diagnostics land, so it is captured and
        // attached to the error on a non-zero exit (an OOM or a bad GGUF is
        // otherwise invisible).
        let mut stderr = String::new();
        let mut raw: Vec<u8> = Vec::new();
        let mut streamed_upto = 0usize;
        let mut completion_start: Option<usize> = None;

        let stdout = child.stdout.take().expect("stdout piped");
        let stderr_handle = child.stderr.take().expect("stderr piped");
        // Stop is meant to be immediate, and a read on llama-cli's stdout can
        // block for the whole model load (tens of seconds on a large GGUF)
        // before the first token makes it back here. So the process is killed
        // by a watcher rather than by the read loop noticing between tokens:
        // killing it closes stdout, which is what unblocks the read.
        let child = Mutex::new(child);
        let finished = AtomicBool::new(false);

        let read_result = std::thread::scope(|scope| {
            let watcher = scope.spawn(|| {
                while !finished.load(Ordering::Relaxed) {
                    if cancel.is_cancelled() {
                        if let Ok(mut child) = child.lock() {
                            let _ = child.kill();
                        }
                        return;
                    }
                    std::thread::sleep(CANCEL_POLL);
                }
            });

            let mut reader = BufReader::new(stdout);
            let mut stderr_reader = BufReader::new(stderr_handle);
            let mut buffer = [0u8; 4096];

            let io = loop {
                if cancel.is_cancelled() {
                    break Ok(false);
                }
                match reader.read(&mut buffer) {
                    Ok(0) => break Ok(true),
                    Ok(n) => {
                        raw.extend_from_slice(&buffer[..n]);
                        // Only the bytes after the echoed prompt are the
                        // completion; stream those the moment they arrive
                        // (`--simple-io` flushes per token), not after a
                        // newline, or the whole summary would batch up.
                        if completion_start.is_none() {
                            completion_start = find_completion_start(&raw, user);
                        }
                        if let Some(start) = completion_start {
                            let from = streamed_upto.max(start);
                            if raw.len() > from {
                                let tail = &raw[from..];
                                // Never stream the timing line / footer to the UI.
                                let cut =
                                    find_subslice(tail, b"\n[ Prompt: ").unwrap_or(tail.len());
                                streamed_upto = from + cut;
                                let usable = &tail[..cut];
                                if !usable.is_empty() {
                                    let text =
                                        String::from_utf8_lossy(usable).into_owned();
                                    if let Some(sink) = on_token {
                                        sink(&text);
                                    }
                                }
                            }
                        }
                    }
                    Err(err) => break Err(err),
                }
            };
            // Drain stderr regardless (the pipe could fill and block the child
            // if we never read it) — best-effort, since only the tail matters.
            let _ = stderr_reader.read_to_string(&mut stderr);
            finished.store(true, Ordering::Relaxed);
            let _ = watcher.join();
            io
        });

        let mut child = child.into_inner().expect("watcher never panics holding the lock");
        let status = child.wait().context("cannot wait for llama-cli")?;

        let completed = match read_result {
            Ok(completed) => completed,
            Err(err) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(SummarizeError::Other(anyhow::Error::new(err)));
            }
        };

        if cancel.is_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(SummarizeError::Cancelled);
        }

        if !status.success() || !completed {
            let detail = stderr.trim();
            return Err(SummarizeError::Other(anyhow::anyhow!(
                "llama-cli failed{}: {detail}",
                if detail.is_empty() { " (no diagnostics)" } else { "" }
            )));
        }

        Ok(extract_completion(&raw, user))
    }
}

/// The echoed-prompt block llama-cli prints before the completion: `\n> ` plus
/// the prompt text, or its first 500 bytes with a truncation marker when the
/// prompt is longer (which every real meeting's transcript is).
fn echo_block(user: &str) -> Vec<u8> {
    let mut block = b"\n> ".to_vec();
    let bytes = user.as_bytes();
    if bytes.len() > 500 {
        block.extend_from_slice(&bytes[..500]);
        block.extend_from_slice(b" ... (truncated)");
    } else {
        block.extend_from_slice(bytes);
    }
    block
}

/// Byte offset of `needle` in `haystack`, or `None`.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack.windows(needle.len()).position(|window| window == needle)
}

/// Byte offset in `raw` where the completion begins — right after the echoed
/// prompt block. `None` until enough of stdout has been read to see it.
///
/// Searched as raw bytes, not text: a truncated prompt can split a multi-byte
/// UTF-8 character, and a lossy string would then report a byte offset that no
/// longer lines up with `raw`.
fn find_completion_start(raw: &[u8], user: &str) -> Option<usize> {
    let block = echo_block(user);
    find_subslice(raw, &block).map(|pos| pos + block.len())
}

/// The completion out of llama-cli's stdout, minus the trailing timing line and
/// "Exiting..." footer. Falls back to the whole output if the echo was never
/// seen (e.g. a different llama-cli version).
fn extract_completion(raw: &[u8], user: &str) -> String {
    let start = find_completion_start(raw, user).unwrap_or(0);
    let mut completion = String::from_utf8_lossy(&raw[start..]).into_owned();
    if let Some(pos) = completion.find("\n[ Prompt: ") {
        completion.truncate(pos);
    }
    if let Some(pos) = completion.rfind("Exiting...") {
        completion.truncate(pos);
    }
    completion.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_args_shape_is_stable() {
        let args = build_args(
            Path::new("/models/gemma.gguf"),
            "system",
            Path::new("/tmp/prompt.txt"),
            4096,
            8,
        );
        let joined = args.join(" ");
        assert!(joined.contains("-m /models/gemma.gguf"));
        assert!(joined.contains("-sys system"));
        assert!(joined.contains("-f /tmp/prompt.txt"));
        assert!(joined.contains("--jinja"));
        assert!(joined.contains("-no-cnv"));
        assert!(joined.contains("-ngl 99"));
        assert!(joined.contains("-c 4096"));
        assert!(joined.contains("-n 2048"));
        assert!(joined.contains("--temp 0.3"));
        assert!(joined.contains("--top-p 0.9"));
        assert!(joined.contains("-t 8"));
        assert!(joined.contains("--no-warmup"));
        assert!(joined.contains("--simple-io"));
        assert!(joined.contains("-st"));
    }

    #[test]
    fn the_completion_is_extracted_from_behind_the_echoed_prompt() {
        // Exactly what v0.2.0's llama-cli writes to stdout: banner, the echoed
        // prompt, the completion, then the timing line and footer.
        let user = "[00:00:01] xin chào\n[00:00:05] quyết định dùng qwen";
        let raw = format!(
            "\n\n▄▄ banner\n\navailable commands:\n  /exit\n\n\n> {user}\n\
             Quyết định: dùng qwen\nHành động: cài đặt\n\n\
             [ Prompt: 924.6 t/s | Generation: 70.1 t/s ]\n\n\nExiting...\n"
        );
        let extracted = extract_completion(raw.as_bytes(), user);
        assert_eq!(extracted, "Quyết định: dùng qwen\nHành động: cài đặt");
    }

    #[test]
    fn a_long_prompt_is_truncated_in_the_echo_and_still_extracted() {
        let user = "x".repeat(600);
        // The echo shows only the first 500 bytes plus the truncation marker.
        let echoed = format!("\n> {} ... (truncated)\n", &user[..500]);
        let raw = format!("banner\n{echoed}the completion text\n[ Prompt: 10 t/s ]\nExiting...");
        let extracted = extract_completion(raw.as_bytes(), &user);
        assert_eq!(extracted, "the completion text");
    }

    #[test]
    fn extraction_tolerates_a_missing_echo() {
        // A different llama-cli that does not echo: fall back to the whole
        // output rather than failing the summary.
        let raw = b"just a completion, no banner".to_vec();
        assert_eq!(extract_completion(&raw, "user"), "just a completion, no banner");
    }

    #[test]
    #[ignore]
    fn e2e_generate_with_a_real_model() {
        use crate::state::CancelFlag;
        let binary = std::path::PathBuf::from(std::env::var("TEST_LLAMA_BIN").expect("TEST_LLAMA_BIN"));
        let model = std::path::PathBuf::from(std::env::var("TEST_LLAMA_MODEL").expect("TEST_LLAMA_MODEL"));
        let summarizer = LlamaCliSummarizer {
            binary,
            model_path: model,
            model_id: "e2e".to_string(),
            context: 4096,
            threads: 4,
        };
        let cancel = CancelFlag::new();
        let streamed = std::sync::Mutex::new(String::new());
        let result = summarizer
            .generate(
                "Bạn là trợ lý tóm tắt cuộc họp.",
                "[00:00:01] Chúng ta quyết định dùng Qwen\n[00:00:05] Hành động: cài đặt",
                Some(&|token| streamed.lock().unwrap().push_str(token)),
                &cancel,
            )
            .unwrap();
        eprintln!("streamed: {:?}", streamed.lock().unwrap());
        eprintln!("result: {result:?}");
        assert!(!result.is_empty(), "a real model must produce a completion");
        assert!(
            !streamed.lock().unwrap().contains("[ Prompt:"),
            "the timing footer must not reach the UI"
        );
    }
}
