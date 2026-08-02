import { useCallback, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { EngineState } from "./EngineControls";
import BatchImportBar from "./BatchImportBar";
import TranscriptPane from "./TranscriptPane";
import SettingsPane from "./SettingsPane";
import { StopIcon, GearIcon, DocIcon } from "./icons";

type Tab = "transcript" | "settings";

export default function App() {
  const [tab, setTab] = useState<Tab>("transcript");
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
    <div className="app-shell">
      <aside className="icon-rail">
        {jobId === null ? (
          <BatchImportBar disabled={false} iconOnly onImport={handleImport} />
        ) : (
          <button
            className="rail-btn primary recording"
            onClick={handleCancel}
            title="Cancel transcription"
            aria-label="Cancel transcription"
          >
            <StopIcon />
          </button>
        )}
        <div className="rail-spacer" />
        <button
          className="rail-btn"
          onClick={() => setTab("settings")}
          title="Settings"
          aria-label="Settings"
        >
          <GearIcon />
        </button>
      </aside>
      <div className="app-main">
        <nav className="pill-nav">
          <button
            className={tab === "transcript" ? "pill active" : "pill"}
            onClick={() => setTab("transcript")}
          >
            <DocIcon /> Transcript
          </button>
          <button
            className={tab === "settings" ? "pill active" : "pill"}
            onClick={() => setTab("settings")}
          >
            <GearIcon /> Settings
          </button>
        </nav>
        <div className="content-card">
          <h1 className="content-title">Meeting Transcriber</h1>
          {tab === "transcript" ? (
            <TranscriptPane running={jobId !== null} onJobDone={handleJobDone} />
          ) : (
            <SettingsPane engine={engine} onEngineChange={setEngine} />
          )}
        </div>
      </div>
    </div>
  );
}
