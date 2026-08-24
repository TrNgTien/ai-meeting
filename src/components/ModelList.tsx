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
 * Shared by the Settings tab, the Manage-models dialog, and the LLM list in
 * the Summary settings. The commands and event names are parameterised because
 * the whisper checkpoints are identified by `name` and the LLM GGUFs by `id`:
 * the two differ only in the command/payload keys, not in the behaviour.
 *
 * The list itself is the caller's single subscription passed down; only download
 * progress (mm_progress / mm_download_finished, see engine.rs) is local.
 *
 * Delete is two-step — the button flips to "Confirm" — because a mis-click
 * costs a multi-GB re-download.
 */
export default function ModelList({
  models,
  active,
  downloadCommand = "download_model",
  deleteCommand = "delete_model",
  cancelCommand = "cancel_download",
  progressEvent = "mm_progress",
  finishedEvent = "mm_download_finished",
  keyField = "name",
  suggested,
}: {
  models: ModelInfo[];
  /** Id of the model the engine is set to; marked so it is not deleted blind. */
  active?: string;
  /** Id of the model to recommend, marked so the accurate default is visible
   * next to the faster-but-worse ones. */
  suggested?: string;
  downloadCommand?: string;
  deleteCommand?: string;
  cancelCommand?: string;
  progressEvent?: string;
  finishedEvent?: string;
  keyField?: "name" | "id";
}) {
  const [progress, setProgress] = useState<Record<string, DownloadProgress>>({});
  const [confirming, setConfirming] = useState<string | null>(null);

  useEffect(() => {
    const unlisten = listen<Record<string, unknown>>("engine-event", (event) => {
      const payload = event.payload;
      switch (payload.event) {
        case progressEvent: {
          const { model, downloaded, total } = payload as unknown as {
            model: string;
            downloaded: number;
            total: number;
          };
          setProgress((prev) => ({ ...prev, [model]: { downloaded, total } }));
          break;
        }
        case finishedEvent: {
          const key = (payload as unknown as Record<string, unknown>)[keyField] as string;
          setProgress((prev) => {
            const next = { ...prev };
            delete next[key];
            return next;
          });
          break;
        }
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, [progressEvent, finishedEvent, keyField]);

  const keyOf = (model: ModelInfo) => (keyField === "id" ? model.name : model.name);

  return (
    <ul className="model-list">
      {models.map((model) => {
        const key = keyOf(model);
        const inFlight = progress[key];
        const pct =
          inFlight && inFlight.total > 0
            ? Math.round((inFlight.downloaded / inFlight.total) * 100)
            : null;
        return (
          <li key={key} className="model-row">
            <div className="model-row-main">
              <span className={`dot ${model.downloaded ? "ready" : "missing"}`} />
              <span className="model-name">{model.name}</span>
              {key === active && <span className="model-badge">in use</span>}
              {key === suggested && <span className="model-badge suggested">suggested</span>}
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
                  <button onClick={() => invoke(cancelCommand, { [keyField]: key })}>
                    Cancel
                  </button>
                </>
              ) : model.downloaded ? (
                confirming === key ? (
                  <>
                    <button
                      className="danger"
                      onClick={() => {
                        invoke(deleteCommand, { [keyField]: key });
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
                    onClick={() => setConfirming(key)}
                  >
                    <TrashIcon />
                  </button>
                )
              ) : (
                <button
                  className="primary"
                  onClick={() => invoke(downloadCommand, { [keyField]: key })}
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
