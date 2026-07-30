/**
 * Placeholder shell. The real UI — header controls, level meters, transcript
 * pane with dimmed preview lines, model manager — lands with the IPC layer.
 */
export default function App() {
  return (
    <main className="shell">
      <h1>Meeting Transcriber</h1>
      <p className="muted">
        Rust + Tauri port in progress. The transcription core is wired up before
        the UI, so this window is intentionally empty for now.
      </p>
    </main>
  );
}
