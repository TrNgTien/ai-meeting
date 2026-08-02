import { useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { UploadIcon } from "./icons";

// Mirrors sidecar.py's AUDIO_EXTS / app.py's AUDIO_EXTS (app.py:56-58).
const AUDIO_EXTS = ["mp3", "wav", "m4a", "aac", "flac", "ogg", "opus", "wma", "mp4"];

function isAudioPath(path: string): boolean {
  const ext = path.split(".").pop()?.toLowerCase();
  return ext !== undefined && AUDIO_EXTS.includes(ext);
}

/** Popup opened from the sidebar's upload icon: a drag-and-drop zone backed
 * by Tauri's window-level file drop event (real file paths, not browser
 * File objects — this isn't a web <input type="file"> drop), plus a native
 * picker as the alternative to dragging.
 */
export default function ImportDialog({
  onClose,
  onImport,
}: {
  onClose: () => void;
  onImport: (paths: string[]) => void;
}) {
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    const unlisten = getCurrentWebview().onDragDropEvent((event) => {
      switch (event.payload.type) {
        case "enter":
        case "over":
          setDragActive(true);
          break;
        case "drop": {
          setDragActive(false);
          const paths = event.payload.paths.filter(isAudioPath);
          if (paths.length > 0) {
            onImport(paths);
            onClose();
          }
          break;
        }
        case "leave":
          setDragActive(false);
          break;
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, [onImport, onClose]);

  async function handleChoose() {
    const selection = await open({
      multiple: true,
      filters: [{ name: "Audio", extensions: AUDIO_EXTS }],
    });
    if (selection === null) return;
    const paths = Array.isArray(selection) ? selection : [selection];
    if (paths.length > 0) {
      onImport(paths);
      onClose();
    }
  }

  return (
    <div className="import-overlay" onClick={onClose}>
      <div className="import-dialog" onClick={(e) => e.stopPropagation()}>
        <div className={dragActive ? "import-dropzone active" : "import-dropzone"}>
          <UploadIcon />
          <p>Drag audio files here</p>
          <p className="muted">or</p>
          <button className="primary" onClick={handleChoose}>
            Choose files…
          </button>
        </div>
        <button className="import-close" onClick={onClose} aria-label="Close">
          Cancel
        </button>
      </div>
    </div>
  );
}
