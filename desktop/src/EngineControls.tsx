import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export interface EngineState {
  langMode: string;
  model: string;
  mlx: boolean;
}

interface ModelInfo {
  name: string;
  downloaded: boolean;
  size_bytes: number;
}

/** Mirrors app.py's header controls (app.py:275-319): language/model
 * dropdowns and the GPU (MLX) switch, default vi+en / first known model / on.
 */
export default function EngineControls({
  value,
  onChange,
}: {
  value: EngineState;
  onChange: (next: EngineState) => void;
}) {
  const [models, setModels] = useState<string[]>([]);

  useEffect(() => {
    invoke("list_models");
    const unlisten = listen<Record<string, unknown>>("sidecar-event", (event) => {
      const payload = event.payload;
      if (payload.event === "models") {
        setModels((payload.models as ModelInfo[]).map((m) => m.name));
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  return (
    <div className="engine-controls">
      <label>
        Language:
        <select
          value={value.langMode}
          onChange={(e) => onChange({ ...value, langMode: e.target.value })}
        >
          <option value="vi+en">vi+en</option>
          <option value="vi">vi</option>
          <option value="en">en</option>
          <option value="auto">auto</option>
        </select>
      </label>
      <label>
        Model:
        <select
          value={value.model}
          onChange={(e) => onChange({ ...value, model: e.target.value })}
        >
          {models.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </label>
      <label>
        <input
          type="checkbox"
          checked={value.mlx}
          onChange={(e) => onChange({ ...value, mlx: e.target.checked })}
        />
        GPU (MLX)
      </label>
    </div>
  );
}
