import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import ImportZone, { pickAudioFiles } from "./ImportZone";
import { StopIcon, UploadIcon } from "./icons";

/** Mirrors app.py's _handle_ui_event (app.py:1360-1477): status line, saved
 * text, and a dimmed live-preview line that's replaced wholesale by the next
 * chunk_text rather than accumulated.
 *
 * There is no "backend disconnected" state to handle: the engine runs in the
 * app's own process, so if it is gone the window is gone with it.
 */
export default function TranscriptPane({
  running,
  onImport,
  onJobDone,
  onCancel,
}: {
  running: boolean;
  onImport: (paths: string[]) => void;
  onJobDone: () => void;
  onCancel: () => void;
}) {
  const [status, setStatus] = useState("");
  const [text, setText] = useState("");
  const [preview, setPreview] = useState("");
  const [stopping, setStopping] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);

  async function handleImportMore() {
    const paths = await pickAudioFiles();
    if (paths.length > 0) onImport(paths);
  }

  // A fresh job (running flips false -> true) supersedes any earlier
  // "Stopping…" state and leftover transcript text from the previous run.
  useEffect(() => {
    if (running) {
      setStopping(false);
      setText("");
      setPreview("");
    }
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
        case "chunk_progress":
          // ETA/progress UI is a follow-up; these are informational only for now.
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
          }
          onJobDone();
          break;
        }
        case "error": {
          const message = payload.message as string;
          const file = payload.file as string | undefined;
          setText((prev) => `${prev}\n[error${file ? ` (${file})` : ""}: ${message}]`);
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
          <span className="running-status">
            {stopping ? "Stopping — finishing the current chunk…" : status || "Starting…"}
          </span>
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
      {!running && text && (
        <div className="status-line">
          <span>{status}</span>
          <button className="import-more-btn" onClick={handleImportMore}>
            <UploadIcon />
            Import more files…
          </button>
        </div>
      )}
      {!running && !text && <div className="status-line">{status}</div>}
      {!running && !text ? (
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
