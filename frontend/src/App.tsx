import { useCallback, useEffect, useState } from "react";
import {
  createPerformance,
  editSegment,
  getHealth,
  getPerformance,
  render,
  type GenerateRequest,
  type Performance,
} from "./api";
import GenerateForm from "./GenerateForm";
import Player from "./Player";
import RatingForm from "./RatingForm";
import ScriptView from "./ScriptView";

const RUNNING = new Set(["analysing", "planning", "writing", "queued", "synthesizing", "rewriting", "modulating", "mixing"]);

function saveJson(name: string, data: unknown) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  a.download = `${name}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

export default function App() {
  const [id, setId] = useState<string | null>(location.hash.slice(1) || null);
  const [perf, setPerf] = useState<Performance | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [serverMock, setServerMock] = useState(false);

  const fail = (e: unknown) => setError(String(e));
  const refresh = useCallback((pid: string) => getPerformance(pid).then(setPerf).catch(fail), []);

  useEffect(() => {
    getHealth().then((h) => setServerMock(h.mock)).catch(fail);
  }, []);

  useEffect(() => {
    if (id) refresh(id);
  }, [id, refresh]);

  const stage = perf?.status.stage;
  useEffect(() => {
    if (!id || !stage || !RUNNING.has(stage)) return;
    const t = setInterval(() => refresh(id), 1500);
    return () => clearInterval(t);
  }, [id, stage, refresh]);

  async function onGenerate(req: GenerateRequest) {
    setError(null);
    setPerf(null);
    try {
      const r = await createPerformance(req);
      location.hash = r.id;
      setId(r.id);
    } catch (e) {
      fail(e);
    }
  }

  const running = stage ? RUNNING.has(stage) : false;

  return (
    <main>
      <h1>Harikatha</h1>
      {serverMock && (
        <p className="warn">
          Mock mode: the server uses placeholder text and a synthetic tone instead of Sarvam. Set MOCK_SARVAM=false in .env
          and restart.
        </p>
      )}
      <GenerateForm busy={running} onSubmit={onGenerate} />
      {error && <p className="error">{error}</p>}

      {perf && (
        <>
          <div className="card">
            <b>{perf.request.title}</b> · {perf.status.stage}
            {perf.status.detail && ` (${perf.status.detail})`}
            {perf.status.script_version && ` · script v${perf.status.script_version}`}
            {perf.status.tts_backend && perf.status.tts_backend !== "mock" && ` · voice: ${perf.status.tts_backend}`}
            {perf.status.progress !== null && RUNNING.has(perf.status.stage) && (
              <progress value={perf.status.progress} max={1} />
            )}
            {perf.mock && <p className="warn">This performance was made in mock mode (placeholder audio).</p>}
            {perf.status.stage === "error" && <p className="error">{perf.status.detail}</p>}
            {perf.script && (
              <div className="row">
                <button disabled={running} onClick={() => render(perf.id).then(() => refresh(perf.id)).catch(fail)}>
                  Render performance
                </button>
              </div>
            )}
          </div>

          {perf.status.stage === "done" && perf.timeline && (
            <>
              <Player
                id={perf.id}
                files={perf.status.files ?? []}
                timeline={perf.timeline}
                credits={perf.credits}
                onActive={setActiveId}
                onDownloadJson={(n) => saveJson(n, n === "script" ? perf.script : perf.timeline)}
              />
              <RatingForm id={perf.id} />
            </>
          )}

          {perf.script && (
            <ScriptView
              segments={perf.script.segments}
              activeId={activeId}
              editable={!running}
              onSave={(sid, edit) => editSegment(perf.id, sid, edit).then(() => refresh(perf.id)).catch(fail)}
            />
          )}
        </>
      )}
    </main>
  );
}
