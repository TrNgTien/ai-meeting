import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export interface ModelInfo {
  name: string;
  downloaded: boolean;
  size_bytes: number;
}

export function formatSize(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
  return `${Math.round(bytes / 1_000_000)} MB`;
}

/** The model list, shared by every pane that shows or picks a checkpoint.
 *
 * `list_models` answers on the emitting thread, so the request must not be
 * sent until `listen` has actually registered — invoking first (as each pane
 * used to do independently) drops the reply and leaves the dropdown empty.
 * One subscription per caller is fine; the point is the ordering.
 */
export function useModels(): { models: ModelInfo[]; refresh: () => void } {
  const [models, setModels] = useState<ModelInfo[]>([]);

  const refresh = useCallback(() => {
    invoke("list_models");
  }, []);

  useEffect(() => {
    let alive = true;
    const unlisten = listen<Record<string, unknown>>("engine-event", (event) => {
      const payload = event.payload;
      if (payload.event === "models") {
        setModels(payload.models as ModelInfo[]);
      }
    });
    unlisten.then(() => {
      if (alive) invoke("list_models");
    });
    return () => {
      alive = false;
      unlisten.then((fn) => fn());
    };
  }, []);

  return { models, refresh };
}
