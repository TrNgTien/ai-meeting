import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import EngineControls, { EngineState } from "./EngineControls";
import TranscriptPane from "./TranscriptPane";
import SettingsPane from "./SettingsPane";
import FilesPane from "./FilesPane";
import ImportDialog from "./ImportDialog";
import ModelDownloadPrompt from "./ModelDownloadPrompt";
import { GearIcon, DocIcon, FolderIcon } from "./icons";

type Tab = "transcript" | "files" | "settings";

interface ModelInfo {
  name: string;
  downloaded: boolean;
  size_bytes: number;
}

export default function App() {
  const [tab, setTab] = useState<Tab>("transcript");
  const [importing, setImporting] = useState(false);
  const [engine, setEngine] = useState<EngineState>({
    langMode: "vi+en",
    model: "small",
    mlx: true,
  });
  const [jobId, setJobId] = useState<string | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [pendingImport, setPendingImport] = useState<string[] | null>(null);
  const [downloadingPending, setDownloadingPending] = useState(false);

  useEffect(() => {
    invoke("list_models");
    const unlisten = listen<Record<string, unknown>>("sidecar-event", (event) => {
      const payload = event.payload;
      if (payload.event === "models") {
        setModels(payload.models as ModelInfo[]);
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  const beginTranscription = useCallback(
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

  const handleImport = useCallback(
    (paths: string[]) => {
      const known = models.find((m) => m.name === engine.model);
      if (known && !known.downloaded) {
        setPendingImport(paths);
        return;
      }
      beginTranscription(paths);
    },
    [models, engine.model, beginTranscription]
  );

  useEffect(() => {
    if (!downloadingPending) return;
    const unlisten = listen<Record<string, unknown>>("sidecar-event", (event) => {
      const payload = event.payload as { event: string } & Record<string, unknown>;
      if (payload.event === "mm_download_finished" && payload.name === engine.model) {
        setDownloadingPending(false);
        if (pendingImport) {
          beginTranscription(pendingImport);
          setPendingImport(null);
        }
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, [downloadingPending, engine.model, pendingImport, beginTranscription]);

  const confirmModelDownload = useCallback(() => {
    setDownloadingPending(true);
    invoke("download_model", { name: engine.model });
  }, [engine.model]);

  const cancelModelDownload = useCallback(() => {
    if (downloadingPending) invoke("cancel_download", { name: engine.model });
    setDownloadingPending(false);
    setPendingImport(null);
  }, [downloadingPending, engine.model]);

  const handleCancel = useCallback(() => {
    if (jobId) invoke("cancel_job", { id: jobId });
  }, [jobId]);

  const handleJobDone = useCallback(() => {
    setJobId(null);
  }, []);

  return (
    <div className="app-shell">
      <aside className="icon-rail">
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
        </nav>
        {tab === "transcript" && (
          <div className="engine-bar">
            <EngineControls value={engine} onChange={setEngine} />
          </div>
        )}
        <div className="content-card">
          <h1 className="content-title">Transcriber</h1>
          {/* All panes stay mounted so TranscriptPane/FilesPane keep
              listening for sidecar events (and keep their accumulated
              state) while another tab is showing, instead of losing
              state/events on every tab switch. */}
          <div className={tab === "transcript" ? "tab-panel" : "tab-panel hidden"}>
            <TranscriptPane
              running={jobId !== null}
              onImport={handleImport}
              onJobDone={handleJobDone}
              onCancel={handleCancel}
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
      {pendingImport && (
        <ModelDownloadPrompt
          model={engine.model}
          downloading={downloadingPending}
          onConfirm={confirmModelDownload}
          onCancel={cancelModelDownload}
        />
      )}
    </div>
  );
}
