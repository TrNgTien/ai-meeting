# Transcriber

A local-only macOS app that records a meeting and turns audio into timestamped,
Vietnamese-first transcripts. Nothing you say ever leaves the machine: the model
downloads on first use and every whisper.cpp inference runs on your Mac's GPU
(Metal).

Built with Rust + Tauri 2 + React. There is no Python and no Homebrew ffmpeg at
runtime — the `.app` is self-contained and portable.

## Features

| Area | State |
| --- | --- |
| Chunked / resumable transcription (`src-tauri/src/chunking/`) | Done, tested. Crash-safe checkpoints survive kills and power loss; recovery truncates and redoes only the interrupted chunk. |
| whisper.cpp engine + Metal (`src-tauri/src/transcribe/whisper_cpp.rs`) | Done. CPU build outside Apple silicon. |
| Hallucination + silence control (`src-tauri/src/transcribe/hallucination.rs`) | Done, tested. |
| Streaming preview + progress (`src-tauri/src/transcribe/whisper_cpp.rs`, `src/lib/progress.ts`) | Done. Segments appear as they decode, per-chunk progress with an ETA. |
| Drag-and-drop import (`src/ImportZone.tsx`) | Done. Window-level drop of `mp3`/`wav`/`m4a`/`aac`/`flac`/`ogg`/`opus`/`wma`/`mp4`, or the native picker; several files at once, each transcribed and saved. |
| Model cache, downloader & manager (`src-tauri/src/transcribe/models.rs`, `src/ModelManagerDialog.tsx`) | Done, tested. Resumable via HTTP Range; download/delete from the UI; importing with a missing model prompts to download it, then resumes the import automatically. |
| Two-track recording (`src-tauri/src/audio/`) | Done, tested. ScreenCaptureKit + loopback fallback, wall-clock-aligned tracks. |
| Transcript merge (`src-tauri/src/merge.rs`) | Done, tested. Interleaves `Me` / `Meeting` by timestamp. |
| Recording in the UI (`src-tauri/src/recording.rs`, `src/RecordBar.tsx`) | Done. Record, meters, device picker, then transcribe both tracks and merge. |
| Files tab (`src/FilesPane.tsx`) | Done. Session list of saved transcripts with Reveal in Finder and inline delete. |
| Settings persistence (`src-tauri/src/settings.rs`) | Done. Language, model, recording sides and microphone survive a relaunch. |
| Bundled ffmpeg/ffprobe (`scripts/build-ffmpeg.sh`) | Done. LGPL, audio-decode only, ~3 MB each; a missing decoder is reported at launch, not on the first file. |
| Bundled llama-cli (`scripts/build-llama.sh`) | Done. MIT, Metal with embedded shaders, for local-LLM summarization. |
| Local-LLM summarization (`src-tauri/src/summarize/`, `src/SummaryPane.tsx`) | Done. A Summarize action on any transcript turns it into `<stem>-summary.md` with the bundled `llama-cli` running a GGUF — same privacy contract as whisper, nothing leaves the machine. Map-reduce over long transcripts, streamed to the Summary tab. |
| Headless CLI (`src-tauri/src/bin/transcribe.rs`) | Done. Same engine as the GUI, no window. |
| **SQLite meeting store** | **Not started.** |

137 tests, `cargo clippy` clean.

## Commands

```bash
make            # setup if needed, then launch the app (the one command)
make setup      # install missing toolchain/deps, fetch pnpm packages, build ffmpeg + llama-cli
make dev        # launch with hot reload
make build      # produce the .app/.dmg
make test       # cargo test
make transcribe FILE=meeting.m4a [MODEL=large-v3] [LANG_MODE=vi+en]   # headless CLI
make release    # version-stamped .dmg in dist-release/
make install    # build and install to /Applications
```

`make setup` is idempotent and `make` re-runs it whenever `package.json` or
`pnpm-lock.yaml` changes.

## Requirements

To **build**: `cmake` (whisper.cpp), `pnpm`, the Rust toolchain, and a C compiler
for ffmpeg. macOS 13+ for system-audio capture. All of it is installed by
`make setup` (except Xcode Command Line Tools).

To **run the built app**: nothing. Models download on first use; ffmpeg and
ffprobe are bundled; the first run asks for microphone and screen-recording
permission and stores them with macOS.

## Where things live

| What | Where |
| --- | --- |
| Recordings + transcripts | `~/Documents/Transcriber/recordings/` |
| Downloaded whisper models | `~/.cache/whisper-cpp/` |
| Downloaded LLM GGUFs | `~/.cache/llama-gguf/` |
| Settings | `~/.config/dev.placepad.transcriber/settings.json` |
| Checkpoints (mid-run) | `<audio>.transcript.partial.json`, next to the audio, deleted on completion |
| Summaries | `<transcript>-summary.md`, next to the transcript |

If you used an older version of the app, `~/.cache/whisper` (openai-whisper
weights), `~/.cache/ai-meeting` and the MLX Hugging Face snapshots in
`~/.cache/huggingface` are dead weight now — safe to delete by hand. The app
never touches them.

### The bundled ffmpeg

Every format the app accepts is decoded by shelling out to ffmpeg, so requiring
`brew install ffmpeg` first was the last thing standing between "download the
app" and "use the app". `scripts/build-ffmpeg.sh` builds the pair from source
during `make setup` and `tauri.conf.json` ships them as `externalBin` sidecars,
which Tauri installs next to the executable inside the `.app`.

Built from source rather than downloaded because **every** ready-made static
macOS build — evermeet, osxexperts, Homebrew — is GPL: they all link x264 to
encode video. This app decodes audio and encodes nothing but raw PCM, so
`--disable-gpl --disable-nonfree --disable-version3` plus an explicit list of the
audio decoders the app's formats imply produces an **LGPL v2.1** build at ~3 MB
each instead of ~80 MB. `--disable-autodetect` keeps it that way by refusing to
link whatever happens to be installed on the build machine.

The script verifies the tarball's SHA-256 and then proves the build can decode to
`s16le` on the exact command line `decode.rs` runs — a build missing a decoder
links perfectly happily and would only fail on the user's first real file.

`decode.rs` looks for the sidecar next to the executable, then falls back to
`PATH`, so `cargo run` and the CLI work against a system ffmpeg.
`TRANSCRIBER_FFMPEG_DIR` overrides both.

### Local-LLM summarization

Any finished transcript — an import or a recording's merged conversation — can
be summarised by pressing the Summarize action, which runs a bundled `llama-cli`
against a GGUF the app downloads on first use and caches in
`~/.cache/llama-gguf/`. Nothing leaves the machine, matching the transcription
contract. A long transcript is split on utterance boundaries, summarised per
part, and the parts are folded into one `<transcript>-summary.md` (map-reduce);
the Summary tab streams tokens as they are produced.

The default model is **Gemma 3n E2B** (Q4_K_M, ~2.8 GB — a ~2B-effective
MatFormer that is *serviceable* at Vietnamese, not great). Two selectable
quantizations ship in the catalog, and a "custom Hugging Face repo + file" field
lets you point it at any instruct GGUF you like. **Qwen3-4B-Instruct GGUF
(`unsloth/Qwen3-4B-Instruct-2507-GGUF`) is a noticeably stronger Vietnamese
alternative** — set it as the custom model, and the GGUF's own chat template is
applied automatically, so no code needs to change. Use only *ungated* GGUFs;
the app's downloader sends no auth token.

`scripts/build-llama.sh` builds `llama-cli` during `make setup` and
`tauri.conf.json` ships it as an `externalBin` sidecar, exactly like ffmpeg.
`TRANSCRIBER_LLAMA_DIR` overrides the resolved binary.

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
- **The bundled ffmpeg and a system ffmpeg decode nearly identically, not
  bit-identically.** They agree bit-for-bit on mp3, m4a, flac, wav and wma;
  opus differs by at most 1 LSB out of 32768 (ffmpeg 7.1 vs 8.1 rounding), which
  is inaudible and not worth chasing in a transcript diff.

## Verifying a change

`make test` (137 tests) covers chunking/checkpoint durability, hallucination
filtering, the merge, the recorder's track alignment, the downloader, and the
summarizer's map-reduce. The end-to-end gates
are real audio: `make transcribe FILE=<a recording>` must produce a clean,
timestamped transcript with no hallucinated outro, and a recording with both
sides enabled must land as two WAVs plus a `-conversation.txt` in
`~/Documents/Transcriber/recordings/`. Summarizing that transcript must produce
a `-summary.md` with real content (which also exercises the bundled `llama-cli`
and its Metal embed).

`make install` + renaming Homebrew's ffmpeg proves self-containment: the
installed app must still transcribe, because the sidecar is doing the decoding.
