import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";

/** Mirrors app.py's _handle_ui_event (app.py:1360-1477): status line, saved
 * text, and a dimmed live-preview line that's replaced wholesale by the next
 * chunk_text rather than accumulated.
 */
export default function TranscriptPane({
  onJobDone,
}: {
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
      <div className="status-line">{status}</div>
      <div ref={boxRef} className="transcript-box card">
        <pre>{text}</pre>
        {preview && <pre className="preview">{preview}</pre>}
      </div>
    </div>
  );
}
