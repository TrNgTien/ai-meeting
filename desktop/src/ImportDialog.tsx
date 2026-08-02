import ImportZone from "./ImportZone";

/** Popup opened from the sidebar's upload icon — wraps ImportZone (also used
 * inline in the Transcript tab's empty state) in a dismissible overlay.
 */
export default function ImportDialog({
  onClose,
  onImport,
}: {
  onClose: () => void;
  onImport: (paths: string[]) => void;
}) {
  return (
    <div className="import-overlay" onClick={onClose}>
      <div className="import-dialog" onClick={(e) => e.stopPropagation()}>
        <ImportZone
          onImport={(paths) => {
            onImport(paths);
            onClose();
          }}
        />
        <button className="import-close" onClick={onClose} aria-label="Close">
          Cancel
        </button>
      </div>
    </div>
  );
}
