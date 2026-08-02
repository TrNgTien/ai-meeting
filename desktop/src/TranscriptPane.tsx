import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import BatchImportBar from "./BatchImportBar";

/** Mirrors app.py's _handle_ui_event (app.py:1360-1477): status line, saved
 * text, and a dimmed live-preview line that's replaced wholesale by the next
 * chunk_text rather than accumulated.
 */
export default function TranscriptPane({
  running,
  onImport,
  onJobDone,
}: {
  running: boolean;
  onImport: (paths: string[]) => void;
  onJobDone: () => void;
}) {
  const [status, setStatus] = useState("");
  const [text, setText] = useState("");
  const [preview, setPreview] = useState("");
  const [disconnected, setDisconnected] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const unlisten = listen<Record<string, unknown>>("sidecar-event", (event) => {
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
          const summary = payload.cancelled
            ? `Stopped after ${took}s. ${count} file(s) finished; partial progress saved.`
            : `Done in ${took}s. Transcribed ${count} file(s).`;
          setStatus(summary);
          setText((prev) => `${prev}\n${summary}`);
          onJobDone();
          break;
        }
        case "error": {
          const message = payload.message as string;
          const file = payload.file as string | undefined;
          setText((prev) => `${prev}\n[error${file ? ` (${file})` : ""}: ${message}]`);
          break;
        }
        case "_sidecar_exited":
          setDisconnected(true);
          break;
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
      {disconnected && (
        <div className="banner-error">Backend disconnected — restart the app.</div>
      )}
      {running && (
        <div className="running-banner">
          <span className="running-dot" />
          {status || "Starting…"}
        </div>
      )}
      {!running && <div className="status-line">{status}</div>}
      {!running && !text ? (
        <div className="transcript-empty">
          <p className="muted">No transcript yet — import an audio file to get started.</p>
          <BatchImportBar disabled={false} onImport={onImport} />
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
