/** Progress and ETA for a running transcription.
 *
 * Port of app.py's live ticker (_tick_live / _update_chunk_progress). The
 * question anyone asks after starting an hour-long file is "how much longer",
 * and the only honest way to answer it is to measure this machine on this file:
 * whisper's speed depends on the model, the audio, and what else is running.
 * So the rate comes from the chunks already done, not from a constant.
 */

export interface ChunkProgress {
  /** Seconds of audio transcribed so far. */
  doneSec: number;
  /** Total seconds of audio, when ffprobe could determine it. */
  totalSec: number | null;
  chunksDone: number;
  /** Where a resumed run picked up, so its ETA is not skewed by audio it did
   * not transcribe this run. */
  baselineSec: number;
  /** Wall-clock ms when this run started producing chunks. */
  startedAt: number;
}

export interface ProgressView {
  /** 0-1, or null when the total length is unknown. */
  fraction: number | null;
  /** e.g. "~4m30s left", or null until there is enough to estimate from. */
  eta: string | null;
  /** e.g. "12m of 45m". */
  position: string | null;
}

export function progressView(progress: ChunkProgress | null, now: number): ProgressView {
  if (!progress || progress.chunksDone === 0) {
    return { fraction: null, eta: null, position: null };
  }

  const { doneSec, totalSec, baselineSec, startedAt } = progress;
  const fraction = totalSec && totalSec > 0 ? Math.min(1, doneSec / totalSec) : null;
  const position = totalSec
    ? `${formatClock(doneSec)} of ${formatClock(totalSec)}`
    : formatClock(doneSec);

  // Only the audio transcribed *this* run has a wall-clock time to divide by;
  // a resumed run's earlier chunks were paid for in a previous one.
  const audioThisRun = doneSec - baselineSec;
  const wallThisRun = (now - startedAt) / 1000;
  if (!totalSec || audioThisRun <= 0 || wallThisRun <= 0) {
    return { fraction, eta: null, position };
  }

  const rate = audioThisRun / wallThisRun;
  const remaining = (totalSec - doneSec) / rate;
  return {
    fraction,
    eta: remaining > 1 ? `~${formatElapsed(remaining)} left` : null,
    position,
  };
}

/** `45s`, `4m32s`, `3h12m` — matching chunking::format_elapsed. */
export function formatElapsed(seconds: number): string {
  const total = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0;
  if (total < 60) return `${total}s`;
  if (total < 3600) return `${Math.floor(total / 60)}m${String(total % 60).padStart(2, "0")}s`;
  return `${Math.floor(total / 3600)}h${String(Math.floor((total % 3600) / 60)).padStart(2, "0")}m`;
}

/** `HH:MM:SS`, matching the timestamps in the transcript itself. */
export function formatClock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const hh = String(Math.floor(total / 3600)).padStart(2, "0");
  const mm = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
  const ss = String(total % 60).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}
