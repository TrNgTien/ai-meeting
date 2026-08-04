import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { ModelInfo, formatSize } from "../lib/models";
import { TrashIcon } from "../icons";

interface DownloadProgress {
  downloaded: number;
  total: number;
}

/** Model rows with download/cancel/delete actions.
 *
 * Shared by the Settings tab and the Manage-models dialog opened from the
 * engine bar, so both show the same in-flight downloads. The list itself is
 * `App`'s single `useModels` subscription passed down; only download progress
 * (mm_progress / mm_download_finished, see engine.rs) is local.
 *
 * Delete is two-step — the button flips to "Confirm" — because a mis-click
 * costs a multi-GB re-download.
 */
export default function ModelList({
  models,
  active,
}: {
  models: ModelInfo[];
  /** Name of the model the engine is set to; marked so it is not deleted blind. */
  active?: string;
}) {
  const [progress, setProgress] = useState<Record<string, DownloadProgress>>({});
  const [confirming, setConfirming] = useState<string | null>(null);

  useEffect(() => {
    const unlisten = listen<Record<string, unknown>>("engine-event", (event) => {
      const payload = event.payload;
      switch (payload.event) {
        case "mm_progress": {
          const { model, downloaded, total } = payload as unknown as {
            model: string;
            downloaded: number;
            total: number;
          };
          setProgress((prev) => ({ ...prev, [model]: { downloaded, total } }));
          break;
        }
        case "mm_download_finished": {
          const { name } = payload as unknown as { name: string };
          setProgress((prev) => {
            const next = { ...prev };
            delete next[name];
            return next;
          });
          break;
        }
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  return (
    <ul className="model-list">
      {models.map((model) => {
        const inFlight = progress[model.name];
        const pct =
          inFlight && inFlight.total > 0
            ? Math.round((inFlight.downloaded / inFlight.total) * 100)
            : null;
        return (
          <li key={model.name} className="model-row">
            <div className="model-row-main">
              <span className={`dot ${model.downloaded ? "ready" : "missing"}`} />
              <span className="model-name">{model.name}</span>
              {model.name === active && <span className="model-badge">in use</span>}
              <span className="model-size">
                {model.downloaded ? formatSize(model.size_bytes) : "not downloaded"}
              </span>
            </div>
            <div className="model-row-actions">
              {inFlight ? (
                <>
                  <div className="download-progress">
                    <progress
                      value={inFlight.downloaded}
                      max={inFlight.total || undefined}
                    />
                    <span className="download-progress-label">
                      {formatSize(inFlight.downloaded)}/{formatSize(inFlight.total)}
                      {pct !== null && ` (${pct}%)`}
                    </span>
                  </div>
                  <button onClick={() => invoke("cancel_download", { name: model.name })}>
                    Cancel
                  </button>
                </>
              ) : model.downloaded ? (
                confirming === model.name ? (
                  <>
                    <button
                      className="danger"
                      onClick={() => {
                        invoke("delete_model", { name: model.name });
                        setConfirming(null);
                      }}
                    >
                      Confirm delete
                    </button>
                    <button onClick={() => setConfirming(null)}>Keep</button>
                  </>
                ) : (
                  <button
                    className="icon-button"
                    title={`Delete ${model.name}`}
                    aria-label={`Delete ${model.name}`}
                    onClick={() => setConfirming(model.name)}
                  >
                    <TrashIcon />
                  </button>
                )
              ) : (
                <button
                  className="primary"
                  onClick={() => invoke("download_model", { name: model.name })}
                >
                  Download
                </button>
              )}
            </div>
          </li>
        );
      })}
      {models.length === 0 && <li className="model-empty">Loading models…</li>}
    </ul>
  );
}
