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
import { RUNNING } from "./stages";
import StatusCard from "./StatusCard";

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
      <header className="masthead">
        <h1 className="brand">Harikatha</h1>
        <p className="tagline">Tell a story, and hear it told.</p>
        <div className="ornament" aria-hidden="true" />
      </header>

      {serverMock && (
        <p className="callout warn">
          Mock mode: the server uses placeholder text and a synthetic tone instead of Sarvam. Set MOCK_SARVAM=false in .env
          and restart.
        </p>
      )}
      <GenerateForm busy={running} onSubmit={onGenerate} />
      {error && (
        <p className="callout error" role="alert">
          {error}
        </p>
      )}

      {perf && (
        <>
          <StatusCard perf={perf} running={running} onRender={() => render(perf.id).then(() => refresh(perf.id)).catch(fail)} />

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
