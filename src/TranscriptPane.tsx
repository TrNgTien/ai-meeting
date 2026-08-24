import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import ImportZone, { pickAudioFiles } from "./ImportZone";
import { RetryIcon, SparklesIcon, StopIcon, UploadIcon } from "./icons";
import { ChunkProgress, progressView } from "./lib/progress";

/** Mirrors app.py's _handle_ui_event (app.py:1360-1477): status line, saved
 * text, and a dimmed live-preview line that's replaced wholesale by the next
 * chunk_text rather than accumulated.
 *
 * There is no "backend disconnected" state to handle: the engine runs in the
 * app's own process, so if it is gone the window is gone with it.
 */
export default function TranscriptPane({
  running,
  recording,
  onImport,
  onJobDone,
  onCancel,
  onSummarize,
  canSummarize,
}: {
  running: boolean;
  /** A meeting is being recorded: importing is blocked, and the empty state
   * would otherwise invite it. */
  recording: boolean;
  onImport: (paths: string[]) => void;
  onJobDone: () => void;
  onCancel: () => void;
  onSummarize: (path: string) => void;
  canSummarize: boolean;
}) {
  const [status, setStatus] = useState("");
  const [text, setText] = useState("");
  const [preview, setPreview] = useState("");
  const [stopping, setStopping] = useState(false);
  // The last run hit an error or was stopped, so re-running it is worth
  // offering next to "Import more files…".
  const [incomplete, setIncomplete] = useState(false);
  const [progress, setProgress] = useState<ChunkProgress | null>(null);
  // Ticks once a second purely so the ETA counts down between chunks, which
  // arrive only every ~5 minutes.
  const [now, setNow] = useState(() => Date.now());
  const baseline = useRef(0);
  const startedAt = useRef(Date.now());
  // The on-disk transcript this pane is showing, so the Summarize action knows
  // what to feed the LLM. Set from merged_text (recordings) and batch_done.
  const transcriptPath = useRef<string | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const view = progressView(progress, now);

  async function handleImportMore() {
    const paths = await pickAudioFiles();
    if (paths.length > 0) onImport(paths);
  }

  /** Back to the empty state — the drag-and-drop zone — instead of straight
   * into the native picker, so a failed run can be replaced by dragging a
   * different file in. */
  function handleReset() {
    setText("");
    setPreview("");
    setStatus("");
    setProgress(null);
    setIncomplete(false);
    transcriptPath.current = null;
  }

  // A fresh job (running flips false -> true) supersedes any earlier
  // "Stopping…" state and leftover transcript text from the previous run.
  useEffect(() => {
    if (running) {
      setStopping(false);
      setIncomplete(false);
      setText("");
      setPreview("");
      setProgress(null);
      baseline.current = 0;
      startedAt.current = Date.now();
      transcriptPath.current = null;
    }
  }, [running]);

  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [running]);

  useEffect(() => {
    const unlisten = listen<Record<string, unknown>>("engine-event", (event) => {
      const payload = event.payload as { event: string } & Record<string, unknown>;
      switch (payload.event) {
        case "status":
          setStatus(payload.message as string);
          break;
        case "file_start":
          setPreview("");
          setText((prev) => (prev ? `${prev}\n===== ${payload.name} =====\n` : `===== ${payload.name} =====\n`));
          break;
        case "chunk_baseline":
          // A resumed run starts partway through the file; remember where, so
          // the ETA divides by the audio this run actually transcribed.
          baseline.current = payload.resume_at_sec as number;
          break;
        case "chunk_progress":
          setProgress({
            doneSec: payload.done_sec as number,
            totalSec: (payload.total_sec as number | null) ?? null,
            chunksDone: payload.chunks_done as number,
            baselineSec: baseline.current,
            startedAt: startedAt.current,
          });
          break;
        case "segment_text":
          setPreview(payload.text as string);
          break;
        case "chunk_text":
          setPreview("");
          setText((prev) => prev + (payload.text as string));
          break;
        case "batch_done": {
          setPreview("");
          const count = payload.count as number;
          const took = Math.round(payload.elapsed_sec as number);
          const cancelled = Boolean(payload.cancelled);
          if (cancelled || count === 0) setIncomplete(true);
          const summary = cancelled
            ? `Stopped after ${took}s. ${count} file(s) finished; partial progress saved.`
            : `Done in ${took}s. Transcribed ${count} file(s).`;
          setStatus(summary);
          if (cancelled && count === 0) {
            // Nothing usable came out of this run — go back to the import
            // screen instead of showing a stub transcript.
            setText("");
          } else {
            setText((prev) => `${prev}\n${summary}`);
            const saved = (payload.saved as string[]) ?? [];
            if (saved.length > 0) transcriptPath.current = saved[saved.length - 1];
          }
          onJobDone();
          break;
        }
        case "rec_started": {
          const backend = payload.system_description as string | undefined;
          setText("");
          setPreview("");
          setStatus(backend ? `Recording — meeting audio via ${backend}.` : "Recording…");
          setText(notes(payload.warnings as string[] | undefined));
          break;
        }
        case "rec_failed":
          setStatus(`Could not start recording: ${payload.message as string}`);
          break;
        case "rec_stopped":
          // Warnings raised while stopping — a side that captured nothing —
          // are the last chance to tell the user something is missing from a
          // meeting they cannot record again.
          setText((prev) => prev + notes(payload.warnings as string[] | undefined));
          break;
        case "merged_text":
          // The interleaved conversation replaces the two per-track transcripts
          // it was built from: same content, in the order it was said.
          setPreview("");
          setText(payload.text as string);
          transcriptPath.current = payload.path as string;
          break;
        case "error": {
          const message = payload.message as string;
          const file = payload.file as string | undefined;
          setText((prev) => `${prev}\n[error${file ? ` (${file})` : ""}: ${message}]`);
          setIncomplete(true);
          break;
        }
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, [onJobDone]);

  useEffect(() => {
    boxRef.current?.scrollTo(0, boxRef.current.scrollHeight);
  }, [text, preview]);

  return (
    <div className="transcript-pane">
      {running && (
        <div className="running-banner">
          <span className="running-dot" />
          <div className="running-body">
            <span className="running-status">
              {stopping ? "Stopping — finishing the current chunk…" : status || "Starting…"}
            </span>
            {!stopping && view.position && (
              <span className="running-progress">
                {[view.position, view.eta, "progress saved"].filter(Boolean).join("   ·   ")}
              </span>
            )}
            {view.fraction !== null && (
              <div className="running-bar">
                <div
                  className="running-bar-fill"
                  style={{ width: `${Math.round(view.fraction * 100)}%` }}
                />
              </div>
            )}
          </div>
          <button
            className="running-stop-btn"
            onClick={() => {
              setStopping(true);
              onCancel();
            }}
            disabled={stopping}
            title="Stop transcription"
            aria-label="Stop transcription"
          >
            <StopIcon />
            {stopping ? "Stopping…" : "Stop"}
          </button>
        </div>
      )}
      {!running && (text || status) && (
        <div className="status-line">
          <span>{status}</span>
          <span className="status-actions">
            {transcriptPath.current && (
              <button
                className="summarize-btn"
                onClick={() => onSummarize(transcriptPath.current!)}
                disabled={!canSummarize}
                title={
                  canSummarize
                    ? "Summarize this transcript with the local LLM"
                    : "Pick an LLM in Settings first"
                }
              >
                <SparklesIcon /> Summarize
              </button>
            )}
            {/* After a run that produced nothing usable, re-running the same
                files would fail the same way: clear the pane back to the
                drop zone so another file can be dragged in. */}
            <button
              className="import-more-btn"
              onClick={incomplete ? handleReset : handleImportMore}
              title={incomplete ? "Clear this run and choose another file" : "Transcribe more files"}
            >
              {incomplete ? <RetryIcon /> : <UploadIcon />}
              {incomplete ? "Try another file…" : "Import more files…"}
            </button>
          </span>
        </div>
      )}
      {!running && !recording && !text ? (
        <div className="transcript-empty">
          <p className="muted">No transcript yet — import an audio file to get started.</p>
          <ImportZone onImport={onImport} />
        </div>
      ) : (
        <div ref={boxRef} className="transcript-box">
          {text
            .split("\n")
            .filter((line) => line.length > 0)
            .map((line, i) => (
              <TranscriptLine key={i} line={line} />
            ))}
          {preview && (
            <div className="transcript-row preview">
              <TranscriptLine line={preview} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Recording warnings, rendered into the transcript as `[note] …` lines the
 * way app.py did — they belong with the meeting they describe, not in a toast
 * that disappears before it is read. */
function notes(warnings: string[] | undefined): string {
  if (!warnings?.length) return "";
  return warnings.map((warning) => `[note] ${warning}`).join("\n") + "\n";
}

// Each real transcript line is "[HH:MM:SS] text" (transcriber.py's
// Segment.format_line); render the timestamp as its own muted column,
// matching the reference design's timestamped-row layout. Separators
// ("===== file ====="), errors, and summaries don't match and fall back to
// a plain row.
const TIMESTAMP_LINE = /^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s(.*)$/;

function TranscriptLine({ line }: { line: string }) {
  const match = line.match(TIMESTAMP_LINE);
  if (match) {
    const [, timestamp, rest] = match;
    return (
      <div className="transcript-row">
        <span className="transcript-timestamp">{timestamp}</span>
        <span className="transcript-text">{rest}</span>
      </div>
    );
  }
  return <div className="transcript-row plain">{line}</div>;
}
