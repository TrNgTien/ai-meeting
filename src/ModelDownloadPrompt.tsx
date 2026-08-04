import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { formatSize } from "./lib/models";

/** Blocks starting a transcription on a model that hasn't been downloaded
 * yet, instead of silently downloading it mid-run (which used to surface as
 * an unexplained "Loading '<model>' onto the GPU..." status with no way to
 * back out). Mirrors the import dialog's overlay styling.
 */
export default function ModelDownloadPrompt({
  model,
  downloading,
  onConfirm,
  onCancel,
}: {
  model: string;
  downloading: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  // "Download it now?" is a different question for 460 MB than for 3 GB, and
  // this is the only moment the answer can change anything.
  const [size, setSize] = useState<number | null>(null);
  useEffect(() => {
    let current = true;
    invoke<number | null>("remote_model_size", { name: model })
      .then((bytes) => current && setSize(bytes ?? null))
      .catch(() => undefined);
    return () => {
      current = false;
    };
  }, [model]);

  return (
    <div className="import-overlay" onClick={downloading ? undefined : onCancel}>
      <div className="import-dialog" onClick={(e) => e.stopPropagation()}>
        <p>
          {downloading
            ? `Downloading model "${model}"… transcription will start automatically once it's ready.`
            : `Model "${model}" hasn't been downloaded yet${
                size ? ` (${formatSize(size)})` : ""
              }. Download it now to start transcribing?`}
        </p>
        {!downloading && (
          <button className="import-primary" onClick={onConfirm}>
            Download &amp; Start
          </button>
        )}
        <button className="import-close" onClick={onCancel} disabled={downloading}>
          Cancel
        </button>
      </div>
    </div>
  );
}
