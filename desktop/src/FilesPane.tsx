import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { revealItemInDir } from "@tauri-apps/plugin-opener";

interface SavedFile {
  path: string;
  name: string;
}

function basename(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

function dirname(path: string): string {
  const parts = path.split(/[\\/]/);
  parts.pop();
  return parts.join("/");
}

/** Every transcript batch_done carries the paths it wrote (sidecar.py's
 * cmd_start_transcription worker). Accumulate them across the session, most
 * recent first, with a Reveal-in-Finder action per file — mirrors app.py's
 * reveal_button/_reveal_saved, but as a persistent list instead of a button
 * that only knows about the last batch.
 */
export default function FilesPane() {
  const [files, setFiles] = useState<SavedFile[]>([]);

  useEffect(() => {
    const unlisten = listen<Record<string, unknown>>("sidecar-event", (event) => {
      const payload = event.payload;
      if (payload.event === "batch_done") {
        const saved = (payload.saved as string[]) ?? [];
        if (saved.length === 0) return;
        setFiles((prev) => {
          const existing = new Set(prev.map((f) => f.path));
          const additions = saved
            .filter((path) => !existing.has(path))
            .map((path) => ({ path, name: basename(path) }));
          return [...additions.reverse(), ...prev];
        });
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  if (files.length === 0) {
    return <p className="muted files-empty">No transcripts yet — import a file to get started.</p>;
  }

  return (
    <div className="files-pane">
      {files.map((file) => (
        <div className="file-row" key={file.path}>
          <div className="file-info">
            <div className="file-name">{file.name}</div>
            <div className="file-path">{dirname(file.path)}</div>
          </div>
          <button onClick={() => revealItemInDir(file.path)}>Reveal in Finder</button>
        </div>
      ))}
    </div>
  );
}
