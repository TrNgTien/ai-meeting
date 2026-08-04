import EngineControls, { EngineState } from "./EngineControls";
import ModelList from "./components/ModelList";
import { ModelInfo } from "./lib/models";

/** Engine controls (language/model) plus the model list — inline instead of a
 * modal, matching the reference design's single-panel-per-tab layout.
 *
 * The list comes from `App`'s single `useModels` subscription; the rows and
 * their download/delete actions live in `components/ModelList`, shared with the
 * dialog the engine bar's "Manage" button opens.
 */
export default function SettingsPane({
  engine,
  models,
  onEngineChange,
}: {
  engine: EngineState;
  models: ModelInfo[];
  onEngineChange: (next: EngineState) => void;
}) {
  return (
    <div className="settings-pane">
      <EngineControls value={engine} models={models} onChange={onEngineChange} />
      <h2 className="section-heading">Models</h2>
      <ModelList models={models} active={engine.model} />
    </div>
  );
}
