import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import EngineControls, { EngineState } from "./EngineControls";
import RecordBar, { RecordOptions } from "./RecordBar";
import TranscriptPane from "./TranscriptPane";
import SettingsPane from "./SettingsPane";
import FilesPane from "./FilesPane";
import ImportDialog from "./ImportDialog";
import ModelManagerDialog from "./ModelManagerDialog";
import ModelDownloadPrompt from "./ModelDownloadPrompt";
import { GearIcon, DocIcon, FolderIcon } from "./icons";
import { useModels } from "./lib/models";
import { useSettings } from "./lib/settings";

type Tab = "transcript" | "files" | "settings";

export default function App() {
  const [tab, setTab] = useState<Tab>("transcript");
  const [importing, setImporting] = useState(false);
  const { settings, loaded: settingsLoaded, update: saveSettings } = useSettings();
  const engine: EngineState = {
    langMode: settings.language_mode,
    model: settings.model,
  };
  const recordOptions: RecordOptions = {
    recordMic: settings.record_mic,
    recordSystem: settings.record_system,
    micDeviceId: settings.mic_device_id,
  };
  const setEngine = useCallback(
    (next: EngineState) =>
      saveSettings({ language_mode: next.langMode, model: next.model }),
    [saveSettings]
  );
  const setRecordOptions = useCallback(
    (next: RecordOptions) =>
      saveSettings({
        record_mic: next.recordMic,
        record_system: next.recordSystem,
        mic_device_id: next.micDeviceId,
      }),
    [saveSettings]
  );
  const [jobId, setJobId] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  // Every format the app accepts is decoded by the bundled ffmpeg, so a damaged
  // install can do nothing at all. Better said once at launch than discovered
  // by an import that fails after the meeting is over.
  const [decoderMissing, setDecoderMissing] = useState(false);
  const { models, refresh: refreshModels } = useModels();
  const [managingModels, setManagingModels] = useState(false);
  const [pendingImport, setPendingImport] = useState<string[] | null>(null);
  const [downloadingPending, setDownloadingPending] = useState(false);

  // Downloading or deleting a checkpoint changes what the pickers should show.
  useEffect(() => {
    const unlisten = listen<Record<string, unknown>>("engine-event", (event) => {
      const kind = event.payload.event;
      if (kind === "mm_download_finished" || kind === "model_deleted") refreshModels();
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, [refreshModels]);

  // Default to something usable: a fresh install may only have whatever the
  // user actually downloaded. Only moves off a model that is not on disk, so an
  // explicit pick is never overridden — and only once the saved settings have
  // arrived, or it would fight the model being restored.
  useEffect(() => {
    if (!settingsLoaded || !models.length) return;
    const current = models.find((m) => m.name === engine.model);
    if (current?.downloaded) return;
    const downloaded = models.find((m) => m.downloaded);
    if (downloaded && downloaded.name !== engine.model) {
      saveSettings({ model: downloaded.name });
    }
  }, [settingsLoaded, models, engine.model, saveSettings]);

  const beginTranscription = useCallback(
    (paths: string[]) => {
      const id = crypto.randomUUID();
      setJobId(id);
      invoke("start_transcription", {
        id,
        paths,
        langMode: engine.langMode,
        model: engine.model,
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
    const unlisten = listen<Record<string, unknown>>("engine-event", (event) => {
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

  // Recording drives its own phase in the backend, so the shell only mirrors
  // it. The transcription that follows a recording is an ordinary job, but its
  // id is minted by the backend from the recording's stem — picking it up here
  // is what lets Stop cancel it like any other.
  useEffect(() => {
    const unlisten = listen<Record<string, unknown>>("engine-event", (event) => {
      const payload = event.payload as { event: string } & Record<string, unknown>;
      switch (payload.event) {
        case "rec_started":
          setRecording(true);
          break;
        case "rec_failed":
          setRecording(false);
          break;
        case "rec_stopped": {
          setRecording(false);
          const tracks = [payload.mic_path, payload.system_path].filter(Boolean);
          if (tracks.length > 0) setJobId(`recording-${payload.stem as string}`);
          break;
        }
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  useEffect(() => {
    invoke<boolean>("ffmpeg_ready")
      .then((ready) => setDecoderMissing(!ready))
      .catch(() => setDecoderMissing(true));
  }, []);

  const handleStartRecording = useCallback((options: RecordOptions) => {
    invoke("start_recording", {
      recordMic: options.recordMic,
      recordSystem: options.recordSystem,
      micDeviceId: options.micDeviceId,
      backend: null,
    });
  }, []);

  const handleStopRecording = useCallback(() => {
    invoke("stop_recording", { langMode: engine.langMode, model: engine.model });
  }, [engine]);

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
            <EngineControls
              value={engine}
              models={models}
              onChange={setEngine}
              onManageModels={() => setManagingModels(true)}
            />
            <RecordBar
              recording={recording}
              busy={jobId !== null}
              options={recordOptions}
              onOptionsChange={setRecordOptions}
              onStart={handleStartRecording}
              onStop={handleStopRecording}
            />
          </div>
        )}
        {decoderMissing && (
          <div className="install-warning" role="alert">
            The audio decoder that ships with this app could not be run, so no file
            can be transcribed. Reinstall the app, or install ffmpeg (
            <code>brew install ffmpeg</code>) to use the system copy instead.
          </div>
        )}
        <div className="content-card">
          <h1 className="content-title">Transcriber</h1>
          {/* All panes stay mounted so TranscriptPane/FilesPane keep
              listening for engine events (and keep their accumulated
              state) while another tab is showing, instead of losing
              state/events on every tab switch. */}
          <div className={tab === "transcript" ? "tab-panel" : "tab-panel hidden"}>
            <TranscriptPane
              running={jobId !== null}
              recording={recording}
              onImport={handleImport}
              onJobDone={handleJobDone}
              onCancel={handleCancel}
            />
          </div>
          <div className={tab === "files" ? "tab-panel" : "tab-panel hidden"}>
            <FilesPane />
          </div>
          <div className={tab === "settings" ? "tab-panel" : "tab-panel hidden"}>
            <SettingsPane engine={engine} models={models} onEngineChange={setEngine} />
          </div>
        </div>
      </div>
      {importing && (
        <ImportDialog onClose={() => setImporting(false)} onImport={handleImport} />
      )}
      {managingModels && (
        <ModelManagerDialog
          models={models}
          active={engine.model}
          onClose={() => setManagingModels(false)}
        />
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
