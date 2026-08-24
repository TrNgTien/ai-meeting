import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";

/** The streamed state of one summarization, rebuilt from the `summary_*`
 * events (engine.rs). Tokens accumulate as they arrive and are replaced
 * wholesale by `summary_done` — the same shape TranscriptPane uses for
 * chunk_text/merged_text. */
export interface SummaryState {
  source: string | null;
  model: string | null;
  stage: "map" | "reduce" | null;
  done: number;
  total: number;
  text: string;
  path: string | null;
  error: string | null;
  running: boolean;
}

const IDLE: SummaryState = {
  source: null,
  model: null,
  stage: null,
  done: 0,
  total: 0,
  text: "",
  path: null,
  error: null,
  running: false,
};

/** One `engine-event` subscription owning the whole summary stream. The pane
 * stays mounted behind `.tab-panel.hidden` so this subscription survives tab
 * switches — exactly like the transcript pane's. */
export function useSummary(): SummaryState {
  const [summary, setSummary] = useState<SummaryState>(IDLE);

  useEffect(() => {
    const unlisten = listen<Record<string, unknown>>("engine-event", (event) => {
      const payload = event.payload as { event: string } & Record<string, unknown>;
      switch (payload.event) {
        case "summary_started":
          setSummary({
            ...IDLE,
            source: payload.source as string,
            model: payload.model as string,
            running: true,
          });
          break;
        case "summary_progress":
          setSummary((prev) => ({
            ...prev,
            stage: payload.stage as "map" | "reduce",
            done: payload.done as number,
            total: payload.total as number,
          }));
          break;
        case "summary_token":
          setSummary((prev) => ({ ...prev, text: prev.text + (payload.text as string) }));
          break;
        case "summary_done":
          setSummary((prev) => ({
            ...prev,
            running: false,
            path: payload.path as string,
            text: payload.text as string,
          }));
          break;
        case "summary_failed":
          setSummary((prev) => ({ ...prev, running: false, error: payload.message as string }));
          break;
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  return summary;
}