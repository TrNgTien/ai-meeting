import { useMemo } from "react";
import Dropdown, { DropdownOption } from "./components/Dropdown";
import { MicIcon, StopIcon } from "./icons";
import { formatElapsed, useInputDevices, useLevels } from "./lib/recording";

export interface RecordOptions {
  recordMic: boolean;
  recordSystem: boolean;
  micDeviceId: string | null;
}

/** Recording a meeting live: the two sides, which microphone, and the meters.
 *
 * Port of app.py's record row (_build_record_row). The two checkboxes are both
 * on by default and either may be turned off, because a meeting is not
 * repeatable — recording one side is better than refusing to record.
 */
export default function RecordBar({
  recording,
  busy,
  options,
  onOptionsChange,
  onStart,
  onStop,
}: {
  recording: boolean;
  /** Transcribing: recording is blocked until it finishes. */
  busy: boolean;
  /** Owned by the shell so the choices can be persisted. */
  options: RecordOptions;
  onOptionsChange: (next: RecordOptions) => void;
  onStart: (options: RecordOptions) => void;
  onStop: () => void;
}) {
  const { recordMic, recordSystem, micDeviceId } = options;
  const { devices } = useInputDevices();
  const levels = useLevels(recording);

  const deviceOptions: DropdownOption[] = useMemo(
    () =>
      devices.map((device) => ({
        value: device.id,
        label: device.name,
        hint: device.is_default ? "system default" : undefined,
      })),
    [devices]
  );

  // Nothing selected means "whatever the system default is", which is also what
  // the backend falls back to when a remembered device is unplugged.
  const selectedDevice =
    micDeviceId ?? devices.find((device) => device.is_default)?.id ?? "";

  const canRecord = (recordMic || recordSystem) && !busy;

  return (
    <div className="record-bar">
      <button
        className={recording ? "record-btn recording" : "record-btn"}
        onClick={() => {
          if (recording) onStop();
          else onStart(options);
        }}
        disabled={!recording && !canRecord}
        title={
          busy
            ? "Wait for the current transcription to finish"
            : recording
              ? "Stop recording and transcribe"
              : "Record this meeting"
        }
      >
        {recording ? <StopIcon /> : <MicIcon />}
        {recording ? "Stop recording" : "Record meeting"}
      </button>

      {recording ? (
        <>
          <span className="record-timer">{formatElapsed(levels.elapsed_sec)}</span>
          <Meter label="Me" level={levels.mic} shown={recordMic} />
          <Meter label="Meeting" level={levels.system} shown={recordSystem} />
        </>
      ) : (
        <>
          <label className="record-check">
            <input
              type="checkbox"
              checked={recordMic}
              onChange={(event) =>
                onOptionsChange({ ...options, recordMic: event.target.checked })
              }
              disabled={busy}
            />
            Me (mic)
          </label>
          <label className="record-check">
            <input
              type="checkbox"
              checked={recordSystem}
              onChange={(event) =>
                onOptionsChange({ ...options, recordSystem: event.target.checked })
              }
              disabled={busy}
            />
            Meeting (system audio)
          </label>
          {recordMic && deviceOptions.length > 0 && (
            <Dropdown
              label="Microphone"
              value={selectedDevice}
              options={deviceOptions}
              placeholder="Default microphone"
              onChange={(id) => onOptionsChange({ ...options, micDeviceId: id })}
            />
          )}
        </>
      )}
    </div>
  );
}

function Meter({ label, level, shown }: { label: string; level: number; shown: boolean }) {
  if (!shown) return null;
  return (
    <div className="record-meter" title={label}>
      <span className="record-meter-label">{label}</span>
      <div className="record-meter-track">
        <div className="record-meter-fill" style={{ width: `${Math.round(level * 100)}%` }} />
      </div>
    </div>
  );
}
