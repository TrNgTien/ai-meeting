import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

export interface InputDevice {
  id: string;
  name: string;
  channels: number;
  sample_rate: number;
  is_default: boolean;
}

export interface LiveLevels {
  mic: number;
  system: number;
  elapsed_sec: number;
  running: boolean;
}

const IDLE: LiveLevels = { mic: 0, system: 0, elapsed_sec: 0, running: false };

/** Meter poll interval, matching app.py's METER_TICK_MS. */
const METER_TICK_MS = 100;

/** How much of the previous reading a meter keeps when the new one is lower.
 * Peaks at 10 Hz are jumpy enough to read as noise; decaying towards silence
 * makes the bar track the voice instead of the sample. From app.py's _tick_meters.
 */
const METER_DECAY = 0.72;

/** Speech peaks well below full scale, so a raw 0-1 peak barely moves the bar.
 * The same 3x the Python meters used. */
const METER_GAIN = 3;

export function useInputDevices(): {
  devices: InputDevice[];
  refresh: () => void;
} {
  const [devices, setDevices] = useState<InputDevice[]>([]);

  const refresh = useCallback(() => {
    invoke<InputDevice[]>("list_input_devices")
      .then(setDevices)
      // A machine with no inputs is a valid state, and so is one where
      // enumeration failed; either way the dropdown just has nothing to offer.
      .catch(() => setDevices([]));
  }, []);

  useEffect(refresh, [refresh]);

  return { devices, refresh };
}

/** Poll the meters while recording.
 *
 * Polling rather than a push event: the levels are decoration on a 100 ms tick,
 * and a dropped poll costs one frame of one bar, where a dropped event would
 * need its own recovery path.
 */
export function useLevels(active: boolean): LiveLevels {
  const [levels, setLevels] = useState<LiveLevels>(IDLE);
  const shown = useRef({ mic: 0, system: 0 });

  useEffect(() => {
    if (!active) {
      shown.current = { mic: 0, system: 0 };
      setLevels(IDLE);
      return;
    }

    let cancelled = false;
    const timer = setInterval(async () => {
      try {
        const next = await invoke<LiveLevels>("recording_levels");
        if (cancelled) return;
        shown.current = {
          mic: Math.max(next.mic, shown.current.mic * METER_DECAY),
          system: Math.max(next.system, shown.current.system * METER_DECAY),
        };
        setLevels({
          ...next,
          mic: Math.min(1, shown.current.mic * METER_GAIN),
          system: Math.min(1, shown.current.system * METER_GAIN),
        });
      } catch {
        // The recording thread is busy opening a backend; the next tick will
        // have a reading.
      }
    }, METER_TICK_MS);

    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [active]);

  return levels;
}

/** `12:34` while recording, growing to `1:02:03` only when it has to. */
export function formatElapsed(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const mm = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
  const ss = String(total % 60).padStart(2, "0");
  const hours = Math.floor(total / 3600);
  return hours > 0 ? `${hours}:${mm}:${ss}` : `${mm}:${ss}`;
}
