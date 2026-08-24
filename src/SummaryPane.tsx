import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { useSummary } from "./lib/summary";
import { DocIcon, RevealIcon, SparklesIcon, StopIcon } from "./icons";

/** The streamed meeting summary: tokens accumulate as they arrive and are
 * replaced wholesale by `summary_done`, mirroring how TranscriptPane handles
 * chunk_text/merged_text. Summarizing itself is triggered from the Transcript
 * and Files panes, which know which transcript is on screen — this pane adds
 * the other way in: "Open…" to re-load a `*-summary.md` written by an earlier
 * session, or "Summarize a file…" to point the LLM at a transcript that is
 * not showing anywhere else (an old `*.txt` from a previous run).
 *
 * Stays mounted behind `.tab-panel.hidden` in App, so its `useSummary`
 * subscription survives tab switches and no tokens are missed.
 */
export default function SummaryPane({
  onCancel,
  onSummarize,
  canSummarize,
}: {
  onCancel: () => void;
  onSummarize: (path: string) => void;
  canSummarize: boolean;
}) {
  const summary = useSummary();
  const [copied, setCopied] = useState(false);
  const [stopping, setStopping] = useState(false);
  // A summary read back from disk via "Open…" — separate from the live
  // stream, since it never went through summary_started/summary_token.
  const [opened, setOpened] = useState<{ path: string; text: string } | null>(null);
  const [openError, setOpenError] = useState<string | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);

  const text = opened ? opened.text : summary.text;
  const path = opened ? opened.path : summary.path;

  // A fresh summary (running flips false -> true) supersedes any leftover
  // "Stopping…" state and an opened file — the live stream takes over the
  // pane, same as TranscriptPane's rule for a fresh transcription job.
  useEffect(() => {
    if (summary.running) {
      setStopping(false);
      setOpened(null);
    }
  }, [summary.running]);

  useEffect(() => {
    boxRef.current?.scrollTo(0, boxRef.current.scrollHeight);
  }, [text]);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard unavailable (no focus/permission); Open-in-Finder is the
      // fallback.
    }
  }

  async function handleOpenSummary() {
    const picked = await open({ multiple: false, filters: [{ name: "Summary", extensions: ["md"] }] });
    if (!picked || Array.isArray(picked)) return;
    try {
      const content = await invoke<string>("read_text_file", { path: picked });
      setOpenError(null);
      setOpened({ path: picked, text: content });
    } catch (err) {
      setOpenError(String(err));
    }
  }

  async function handleOpenTranscript() {
    const picked = await open({ multiple: false, filters: [{ name: "Transcript", extensions: ["txt"] }] });
    if (!picked || Array.isArray(picked)) return;
    onSummarize(picked);
  }

  const openActions = (
    <div className="summary-open-actions">
      <button className="import-more-btn" onClick={handleOpenSummary} title="Open a saved summary file">
        <DocIcon />
        Open summary…
      </button>
      <button
        className="import-more-btn"
        onClick={handleOpenTranscript}
        disabled={!canSummarize}
        title={
          canSummarize
            ? "Pick a transcript to summarize"
            : "Pick an LLM in Settings first"
        }
      >
        <SparklesIcon />
        Summarize a file…
      </button>
    </div>
  );

  if (!summary.running && !text) {
    return (
      <div className="summary-empty">
        <p className="muted">
          No summary yet — press <SparklesIcon className="inline-icon" /> Summarize on a
          transcript or a recording to get a local-LLM meeting summary.
        </p>
        {openError && <div className="summary-error">{openError}</div>}
        {openActions}
      </div>
    );
  }

  return (
    <div className="summary-pane">
      {summary.running && (
        <div className="summary-status">
          <span className="running-dot" />
          <span>
            {stopping
              ? "Stopping…"
              : summary.stage === "map" && summary.total > 1
              ? `Reading part ${Math.min(summary.done + 1, summary.total)} of ${summary.total}…`
              : summary.stage === "reduce"
              ? "Weaving the parts together…"
              : "Summarising…"}
          </span>
          <button
            className="running-stop-btn"
            onClick={() => {
              setStopping(true);
              onCancel();
            }}
            disabled={stopping}
            title="Stop summarizing"
            aria-label="Stop summarizing"
          >
            <StopIcon />
            {stopping ? "Stopping…" : "Stop"}
          </button>
        </div>
      )}
      {summary.error && <div className="summary-error">{summary.error}</div>}
      {openError && <div className="summary-error">{openError}</div>}
      <div ref={boxRef} className="summary-box">
        {text}
      </div>
      {path && !summary.running && (
        <div className="summary-actions">
          <button className="icon-button" onClick={copy} title="Copy summary">
            {copied ? "Copied" : "Copy"}
          </button>
          <button
            className="icon-btn"
            onClick={() => revealItemInDir(path)}
            title="Reveal in Finder"
            aria-label="Reveal in Finder"
          >
            <RevealIcon />
          </button>
          {openActions}
        </div>
      )}
    </div>
  );
}
