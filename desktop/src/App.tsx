import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

/**
 * Scaffolding, not product: proves the sidecar pipe end-to-end (Rust spawns
 * sidecar.py -> stdout JSON lines -> Tauri events -> this listener) before
 * any real UI is built on top of it. The real UI — header controls, level
 * meters, transcript pane, model manager — replaces this.
 */
export default function App() {
  const [events, setEvents] = useState<string[]>([]);
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    const unlisten = listen<unknown>("sidecar-event", (event) => {
      setEvents((prev) => [...prev, JSON.stringify(event.payload)]);
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight);
  }, [events]);

  return (
    <main className="shell">
      <h1>Meeting Transcriber</h1>
      <p className="muted">
        Rust + Tauri port in progress. This view proves the sidecar.py pipe
        works; the real UI lands next.
      </p>
      <button onClick={() => invoke("list_models")}>List models</button>
      <pre ref={logRef} style={{ maxHeight: "60vh", overflow: "auto" }}>
        {events.join("\n")}
      </pre>
    </main>
  );
}
