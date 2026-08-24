import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

/** An LLM from the summarize catalog (`list_llm_models`). Identified by `id`,
 * unlike the whisper models' `name` — the ModelList component is parameterised
 * to serve both. */
export interface LlmModelInfo {
  id: string;
  label: string;
  downloaded: boolean;
  size_bytes: number;
}

/** The LLM catalog list, shared by the Summary settings.
 *
 * Same subscription shape as `useModels` — `list_llm_models` answers on the
 * emitting thread, so the invoke must wait for `listen` to register, and the
 * ordering fix at models.ts:38-40 is kept for the same reason.
 */
export function useLlmModels(): { models: LlmModelInfo[]; refresh: () => void } {
  const [models, setModels] = useState<LlmModelInfo[]>([]);

  const refresh = useCallback(() => {
    invoke("list_llm_models");
  }, []);

  useEffect(() => {
    let alive = true;
    const unlisten = listen<Record<string, unknown>>("engine-event", (event) => {
      const payload = event.payload;
      if (payload.event === "llm_models") {
        setModels(payload.models as LlmModelInfo[]);
      }
    });
    unlisten.then(() => {
      if (alive) invoke("list_llm_models");
    });
    return () => {
      alive = false;
      unlisten.then((fn) => fn());
    };
  }, []);

  return { models, refresh };
}