import ModelList from "./components/ModelList";
import { ModelInfo } from "./lib/models";

/** Popup opened from the engine bar's "Manage" button — same model rows as the
 * Settings tab, reachable without leaving the Transcript view.
 *
 * Downloads keep running if the dialog is closed; progress is re-read from the
 * engine events when it is reopened.
 */
export default function ModelManagerDialog({
  models,
  active,
  onClose,
}: {
  models: ModelInfo[];
  active?: string;
  onClose: () => void;
}) {
  return (
    <div className="import-overlay" onClick={onClose}>
      <div
        className="model-dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Manage models"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="model-dialog-title">Manage models</h2>
        <ModelList models={models} active={active} />
        <button className="import-close" onClick={onClose}>
          Done
        </button>
      </div>
    </div>
  );
}
