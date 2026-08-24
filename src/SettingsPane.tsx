import EngineControls, { EngineState } from "./EngineControls";
import ModelList from "./components/ModelList";
import Dropdown, { DropdownOption } from "./components/Dropdown";
import { ModelInfo, SUGGESTED_MODEL, formatSize } from "./lib/models";
import { LlmModelInfo } from "./lib/llm";
import { Settings } from "./lib/settings";

/** Engine controls (language/model) plus the model list — inline instead of a
 * modal, matching the reference design's single-panel-per-tab layout.
 *
 * The whisper list comes from `App`'s single `useModels` subscription; the LLM
 * list from `useLlmModels`. The rows and their download/delete actions live in
 * `components/ModelList`, shared with the dialog the engine bar's "Manage"
 * button opens.
 */
export default function SettingsPane({
  settings,
  models,
  llmModels,
  update,
}: {
  settings: Settings;
  models: ModelInfo[];
  llmModels: LlmModelInfo[];
  update: (patch: Partial<Settings>) => void;
}) {
  const engine: EngineState = { langMode: settings.language_mode, model: settings.model };
  const onEngineChange = (next: EngineState) =>
    update({ language_mode: next.langMode, model: next.model });

  const llmOptions: DropdownOption[] = [
    ...llmModels.map((model) => ({
      value: model.id,
      label: model.label,
      hint: model.downloaded ? formatSize(model.size_bytes) : "not downloaded",
      status: model.downloaded ? ("ready" as const) : ("missing" as const),
    })),
    { value: "custom", label: "Custom (any Hugging Face repo)" },
  ];

  const custom = settings.llm_model === "custom";

  return (
    <div className="settings-pane">
      <EngineControls value={engine} models={models} onChange={onEngineChange} />
      <h2 className="section-heading">Models</h2>
      <ModelList models={models} active={engine.model} suggested={SUGGESTED_MODEL} />

      <h2 className="section-heading">Summary (local LLM)</h2>
      <div className="control">
        <span className="control-label">LLM model</span>
        <Dropdown
          label="LLM model"
          value={settings.llm_model}
          options={llmOptions}
          placeholder={llmModels.length ? "Select an LLM" : "Loading…"}
          onChange={(llm_model) => update({ llm_model })}
        />
      </div>
      {custom && (
        <div className="custom-llm">
          <div className="control">
            <span className="control-label">Hugging Face repo</span>
            <input
              className="text-input"
              type="text"
              value={settings.llm_custom_repo ?? ""}
              placeholder="unsloth/Qwen3-4B-Instruct-2507-GGUF"
              onChange={(e) => update({ llm_custom_repo: e.target.value || null })}
            />
          </div>
          <div className="control">
            <span className="control-label">GGUF file</span>
            <input
              className="text-input"
              type="text"
              value={settings.llm_custom_file ?? ""}
              placeholder="Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
              onChange={(e) => update({ llm_custom_file: e.target.value || null })}
            />
          </div>
          <p className="hint-text">
            The repo must be ungated and the file must end in <code>.gguf</code>. It is
            downloaded on first summarise.
          </p>
        </div>
      )}
      <ModelList
        models={llmModels.map((m) => ({
          name: m.id,
          downloaded: m.downloaded,
          size_bytes: m.size_bytes,
        }))}
        active={settings.llm_model !== "custom" ? settings.llm_model : undefined}
        downloadCommand="download_llm_model"
        deleteCommand="delete_llm_model"
        cancelCommand="cancel_llm_download"
        progressEvent="llm_progress"
        finishedEvent="llm_download_finished"
        keyField="id"
      />
      <label className="toggle-row">
        <input
          type="checkbox"
          checked={settings.summarize_after_recording}
          onChange={(e) => update({ summarize_after_recording: e.target.checked })}
        />
        Summarise a recording automatically after it finishes transcribing
      </label>
    </div>
  );
}