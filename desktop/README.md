# desktop/ — Rust + Tauri port

A port of the Python app at the repo root, structured like
[meetily](https://github.com/Zackriya-Solutions/meetily). The **engines** changed;
the **behaviour** is ported deliberately faithfully.

## Status

| Area | State |
| --- | --- |
| Chunked / resumable transcription (`chunking/`) | Done, tested. Chunk boundaries verified byte-identical to the Python app. |
| whisper.cpp engine + Metal (`transcribe/whisper_cpp.rs`) | Done. A/B'd against Python on the same model. |
| Hallucination + silence control (`transcribe/hallucination.rs`) | Done, tested. |
| Model cache & downloader (`transcribe/models.rs`) | Done, tested. Resumable via HTTP Range. |
| Two-track recording (`audio/`) | Done, tested. ScreenCaptureKit + loopback fallback. |
| Transcript merge (`merge.rs`) | Done, tested. |
| **Tauri commands / events / React UI** | **Not started.** The window opens but is a placeholder. |
| **`vi` mode (PhoWhisper)** | **Not started.** Needs an HF-safetensors → GGML converter. |
| **SQLite meeting store** | **Not started.** |

102 tests, `cargo clippy` clean.

## Commands

Run from the repo root:

```bash
make desktop-setup              # pnpm install
make desktop-test               # cargo test
make desktop-dev                # pnpm tauri dev  (placeholder UI until the IPC layer lands)
make desktop-parity FILE=data/x.mp3   # prove chunk decode/split matches the Python app
```

Headless transcription, which is also the A/B harness:

```bash
cd desktop/src-tauri
cargo run --release --example transcribe -- <audio> [model] [vi+en|en|auto] [chunk_sec]
```

Compare against the Python side with
`.venv/bin/python desktop/scripts/ab_python.py <audio> <model> vi+en <chunk_sec>`.

## Requirements

`ffmpeg` (chunk decoding), `cmake` (whisper.cpp), `pnpm`, and the Rust toolchain.
macOS 13+ for system-audio capture.

## Things that will bite you

- **`whisper-rs` leaks its segment callback.** It stores the closure with
  `Box::into_raw` and implements no `Drop` for `FullParams`, so a channel sender
  captured by that closure is *never dropped*. Waiting for the channel to close
  deadlocks. `whisper_cpp.rs` sends an explicit `None` sentinel instead.
- **`.cargo/config.toml` sets `http.multiplexing = false`.** Without it, cargo's
  LibreSSL intermittently fails the TLS handshake against some crates.io CDN
  nodes and reports a bogus "failed to download".
- **`screencapturekit` is pinned to 2.1 on purpose.** 3.0+ depend on
  `apple-metal`, whose Swift bridge needs a newer Metal SDK than macOS 15
  Command Line Tools ships (`MTLSamplerReductionMode` is not in scope). 2.1 is
  the last version built on objc2.
- **Debug artifacts are ~5 GB.** `target/debug` plus a 1.6 GB model will fill a
  nearly-full disk; `rm -rf target/debug` is the quick win.
- **Model caches are separate from the Python app's.** GGML lives in
  `~/.cache/whisper-cpp`; the Python app's `.pt` files stay in `~/.cache/whisper`
  and are never touched, so both apps keep working.
- **Checkpoints are not shared between the two apps.** `CHECKPOINT_VERSION` is 4
  here vs 3 there, and the engine key differs, so neither can resume into the
  other's transcript — which is the point.

## Verifying a change to the transcription path

The A/B diff is the real gate. Timestamps drift slightly (different decoder);
segment ordering, content, and Vietnamese/English word choice should not. On a
90 s Vietnamese clip with `large-v3-turbo`, several lines come out byte-identical
and every English term (`framework`, `tool`, `library`, `agent`, `CV`, …) is
preserved on both sides — that last part is the whole point of `vi+en` mode.
