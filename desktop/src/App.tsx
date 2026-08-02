import { useCallback, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import ModelManagerDialog from "./ModelManagerDialog";
import EngineControls, { EngineState } from "./EngineControls";
import BatchImportBar from "./BatchImportBar";
import TranscriptPane from "./TranscriptPane";

export default function App() {
  const [managingModels, setManagingModels] = useState(false);
  const [engine, setEngine] = useState<EngineState>({
    langMode: "vi+en",
    model: "small",
    mlx: true,
  });
  const [jobId, setJobId] = useState<string | null>(null);

  const handleImport = useCallback(
    (paths: string[]) => {
      const id = crypto.randomUUID();
      setJobId(id);
      invoke("start_transcription", {
        id,
        paths,
        langMode: engine.langMode,
        model: engine.model,
        mlx: engine.mlx,
      });
    },
    [engine]
  );

  const handleCancel = useCallback(() => {
    if (jobId) invoke("cancel_job", { id: jobId });
  }, [jobId]);

  const handleJobDone = useCallback(() => {
    setJobId(null);
  }, []);

  return (
    <main className="shell">
      <div className="app-header">
        <h1>Meeting Transcriber</h1>
        <p className="muted">Local, offline, Vietnamese-first meeting transcription.</p>
      </div>
      <div className="card">
        <EngineControls value={engine} onChange={setEngine} />
        <div className="toolbar">
          <BatchImportBar disabled={jobId !== null} onImport={handleImport} />
          <button className="destructive" disabled={jobId === null} onClick={handleCancel}>
            Cancel
          </button>
          <button onClick={() => setManagingModels(true)}>Manage models</button>
        </div>
      </div>
      <TranscriptPane onJobDone={handleJobDone} />
      {managingModels && <ModelManagerDialog onClose={() => setManagingModels(false)} />}
    </main>
  );
}
