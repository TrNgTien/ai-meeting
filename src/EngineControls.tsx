import Dropdown, { DropdownOption } from "./components/Dropdown";
import { ModelInfo, formatSize } from "./lib/models";
import { SlidersIcon } from "./icons";

export interface EngineState {
  langMode: string;
  model: string;
}

const LANGUAGES: DropdownOption[] = [
  { value: "vi+en", label: "vi+en", hint: "Vietnamese + English" },
  { value: "en", label: "en", hint: "English" },
  { value: "auto", label: "auto", hint: "Detect" },
];

/** Language and model dropdowns.
 *
 * The model list is owned by `App` (one `useModels` subscription) and passed
 * in, so every pane shows the same state and the list survives tab switches.
 *
 * The Python app's third control, a GPU (MLX) switch, is gone: whisper.cpp
 * compiles Metal in and falls back to CPU inside the library, so there is no
 * runtime choice left to offer. Its `vi` (PhoWhisper) language option is gone
 * for the same kind of reason — see `state::LanguageMode`.
 */
export default function EngineControls({
  value,
  models,
  onChange,
  onManageModels,
}: {
  value: EngineState;
  models: ModelInfo[];
  onChange: (next: EngineState) => void;
  /** Omitted where a model list is already on screen (the Settings tab). */
  onManageModels?: () => void;
}) {
  const modelOptions: DropdownOption[] = models.map((model) => ({
    value: model.name,
    label: model.name,
    hint: model.downloaded ? formatSize(model.size_bytes) : "not downloaded",
    status: model.downloaded ? "ready" : "missing",
  }));

  return (
    <div className="engine-controls">
      <div className="control">
        <span className="control-label">Language</span>
        <Dropdown
          label="Language"
          value={value.langMode}
          options={LANGUAGES}
          onChange={(langMode) => onChange({ ...value, langMode })}
        />
      </div>
      <div className="control">
        <span className="control-label">Model</span>
        <Dropdown
          label="Model"
          value={value.model}
          options={modelOptions}
          placeholder={models.length ? "Select a model" : "Loading…"}
          onChange={(model) => onChange({ ...value, model })}
        />
      </div>
      {onManageModels && (
        <button className="manage-models" onClick={onManageModels}>
          <SlidersIcon /> Manage
        </button>
      )}
    </div>
  );
}
