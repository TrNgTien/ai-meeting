import { useCallback, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { EngineState } from "./EngineControls";
import TranscriptPane from "./TranscriptPane";
import SettingsPane from "./SettingsPane";
import FilesPane from "./FilesPane";
import ImportDialog from "./ImportDialog";
import { UploadIcon, StopIcon, GearIcon, DocIcon, FolderIcon } from "./icons";

type Tab = "transcript" | "files" | "settings";

export default function App() {
  const [tab, setTab] = useState<Tab>("transcript");
  const [importing, setImporting] = useState(false);
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
          <button
            className="rail-btn primary"
            onClick={() => setImporting(true)}
            title="Import files…"
            aria-label="Import files"
          >
            <UploadIcon />
          </button>
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
            className={tab === "files" ? "pill active" : "pill"}
            onClick={() => setTab("files")}
          >
            <FolderIcon /> Files
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
          {/* All panes stay mounted so TranscriptPane/FilesPane keep
              listening for sidecar events (and keep their accumulated
              state) while another tab is showing, instead of losing
              state/events on every tab switch. */}
          <div className={tab === "transcript" ? "tab-panel" : "tab-panel hidden"}>
            <TranscriptPane
              running={jobId !== null}
              onImport={handleImport}
              onJobDone={handleJobDone}
            />
          </div>
          <div className={tab === "files" ? "tab-panel" : "tab-panel hidden"}>
            <FilesPane />
          </div>
          <div className={tab === "settings" ? "tab-panel" : "tab-panel hidden"}>
            <SettingsPane engine={engine} onEngineChange={setEngine} />
          </div>
        </div>
      </div>
      {importing && (
        <ImportDialog onClose={() => setImporting(false)} onImport={handleImport} />
      )}
    </div>
  );
}
