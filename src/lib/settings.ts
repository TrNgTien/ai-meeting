import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

export interface Settings {
  language_mode: string;
  model: string;
  record_mic: boolean;
  record_system: boolean;
  mic_device_id: string | null;
}

const DEFAULTS: Settings = {
  language_mode: "vi+en",
  model: "large-v3",
  record_mic: true,
  record_system: true,
  mic_device_id: null,
};

/** The choices that survive a relaunch.
 *
 * `loaded` matters to callers: until the saved settings arrive, what is on
 * screen is the default, and writing that back would overwrite the user's real
 * choice with a default they never picked.
 */
export function useSettings(): {
  settings: Settings;
  loaded: boolean;
  update: (patch: Partial<Settings>) => void;
} {
  const [settings, setSettings] = useState<Settings>(DEFAULTS);
  const [loaded, setLoaded] = useState(false);
  const latest = useRef(settings);
  latest.current = settings;

  useEffect(() => {
    invoke<Settings>("load_settings")
      .then((stored) => setSettings({ ...DEFAULTS, ...stored }))
      // Unreadable settings are the default settings; the backend already
      // decided that, and the next save fixes the file.
      .catch(() => undefined)
      .finally(() => setLoaded(true));
  }, []);

  const update = useCallback((patch: Partial<Settings>) => {
    const next = { ...latest.current, ...patch };
    latest.current = next;
    setSettings(next);
    invoke("save_settings", { settings: next }).catch(() => undefined);
  }, []);

  return { settings, loaded, update };
}
