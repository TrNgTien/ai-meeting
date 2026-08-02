import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

interface ModelInfo {
  name: string;
  downloaded: boolean;
  size_bytes: number;
}

interface DownloadProgress {
  downloaded: number;
  total: number;
}

/** Mirrors sidecar.py's cmd_list_models/cmd_download_model/cmd_delete_model
 * event shapes exactly (models / mm_progress / mm_download_finished /
 * model_deleted) — see sidecar.py, not the design doc's illustrative names.
 */
export default function ModelManagerDialog({ onClose }: { onClose: () => void }) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [progress, setProgress] = useState<Record<string, DownloadProgress>>({});

  useEffect(() => {
    invoke("list_models");
    const unlisten = listen<Record<string, unknown>>("sidecar-event", (event) => {
      const payload = event.payload;
      switch (payload.event) {
        case "models":
          setModels(payload.models as ModelInfo[]);
          break;
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
          invoke("list_models");
          break;
        }
        case "model_deleted":
          invoke("list_models");
          break;
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  return (
    <div className="model-manager-overlay">
      <div className="model-manager">
        <div className="model-manager-header">
          <h2>Manage models</h2>
          <button onClick={onClose}>Close</button>
        </div>
        <table>
          <tbody>
            {models.map((model) => {
              const inFlight = progress[model.name];
              return (
                <tr key={model.name}>
                  <td>{model.name}</td>
                  <td>
                    {inFlight ? (
                      <progress
                        value={inFlight.downloaded}
                        max={inFlight.total || undefined}
                      />
                    ) : model.downloaded ? (
                      `${(model.size_bytes / 1_000_000).toFixed(0)} MB`
                    ) : (
                      "not downloaded"
                    )}
                  </td>
                  <td>
                    {inFlight ? (
                      <button onClick={() => invoke("cancel_download", { name: model.name })}>
                        Cancel
                      </button>
                    ) : model.downloaded ? (
                      <button onClick={() => invoke("delete_model", { name: model.name })}>
                        Delete
                      </button>
                    ) : (
                      <button onClick={() => invoke("download_model", { name: model.name })}>
                        Download
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
