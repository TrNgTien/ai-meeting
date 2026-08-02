import { open } from "@tauri-apps/plugin-dialog";
import { UploadIcon } from "./icons";

// Mirrors sidecar.py's AUDIO_EXTS / app.py's AUDIO_EXTS (app.py:56-58).
const AUDIO_EXTS = ["mp3", "wav", "m4a", "aac", "flac", "ogg", "opus", "wma", "mp4"];

export default function BatchImportBar({
  disabled,
  iconOnly,
  onImport,
}: {
  disabled: boolean;
  iconOnly?: boolean;
  onImport: (paths: string[]) => void;
}) {
  async function handleClick() {
    const selection = await open({
      multiple: true,
      filters: [{ name: "Audio", extensions: AUDIO_EXTS }],
    });
    if (selection === null) return;
    const paths = Array.isArray(selection) ? selection : [selection];
    if (paths.length > 0) onImport(paths);
  }

  if (iconOnly) {
    return (
      <button
        className="rail-btn primary"
        disabled={disabled}
        onClick={handleClick}
        title="Import files…"
        aria-label="Import files"
      >
        <UploadIcon />
      </button>
    );
  }

  return (
    <button className="primary" disabled={disabled} onClick={handleClick}>
      Import files…
    </button>
  );
}
