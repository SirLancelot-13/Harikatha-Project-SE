import type { Performance } from "./api";
import { RENDER_STEPS, RUNNING, SCRIPT_STEPS, STAGE_HINT } from "./stages";

type StepState = "done" | "current" | "todo";

/** One row of steps. Steps before the current one are done; the other phase is all done or all todo. */
function Steps({ label, steps, stage, otherPhaseDone }: { label: string; steps: readonly string[]; stage: string; otherPhaseDone: boolean }) {
  const at = steps.indexOf(stage);
  const state = (i: number): StepState => (at === -1 ? (otherPhaseDone ? "done" : "todo") : i < at ? "done" : i === at ? "current" : "todo");
  return (
    <div className="steps-group">
      <span className="steps-label">{label}</span>
      <ol className="steps">
        {steps.map((s, i) => (
          <li key={s} className={`step ${state(i)}`} aria-current={state(i) === "current" ? "step" : undefined}>
            <span className="step-dot" aria-hidden="true" />
            {s}
          </li>
        ))}
      </ol>
    </div>
  );
}

export default function StatusCard({
  perf,
  running,
  onRender,
}: {
  perf: Performance;
  running: boolean;
  onRender: () => void;
}) {
  const { status } = perf;
  const stage = status.stage;
  const pct = status.progress !== null ? Math.round(status.progress * 100) : null;
  const voice = status.tts_backend && status.tts_backend !== "mock" ? status.tts_backend : null;
  const failed = stage === "error";
  const scriptDone = perf.script !== null; // a failed render still has its finished script

  return (
    <section className="card status" aria-live="polite">
      <header className="status-head">
        <h2 className="status-title">{perf.request.title}</h2>
        <div className="pills">
          <span className={`pill stage-${stage}${running ? " running" : ""}`}>
            {stage}
            {status.detail && ` (${status.detail})`}
          </span>
          {status.script_version && <span className="pill">script v{status.script_version}</span>}
          {voice && <span className="pill">voice: {voice}</span>}
        </div>
      </header>

      <p className="hint stage-hint">{STAGE_HINT[stage] ?? ""}</p>

      {status.progress !== null && RUNNING.has(stage) && (
        <div className="progress-wrap">
          <progress value={status.progress} max={1} aria-label="Progress" />
          <span className="progress-pct">{pct}%</span>
        </div>
      )}

      <div className="steps-wrap">
        <Steps label="Script" steps={SCRIPT_STEPS} stage={stage} otherPhaseDone={scriptDone} />
        <Steps label="Render" steps={RENDER_STEPS} stage={stage} otherPhaseDone={false} />
      </div>

      {perf.mock && <p className="callout warn">This performance was made in mock mode (placeholder audio).</p>}
      {failed && (
        <p className="callout error" role="alert">
          <strong>Something went wrong.</strong> {status.detail}
        </p>
      )}

      {perf.script && (
        <div className="row">
          <button className="btn primary" disabled={running} onClick={onRender}>
            Render performance
          </button>
          {running && <span className="hint">Working on it. You can keep reading the script meanwhile.</span>}
        </div>
      )}
    </section>
  );
}
