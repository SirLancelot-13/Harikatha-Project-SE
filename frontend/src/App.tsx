import { useCallback, useEffect, useState } from "react";
import {
  createPerformance, editSegment, getHealth, getPerformance, render,
  type GenerateRequest, type Performance,
} from "./api";
import GenerateForm from "./GenerateForm";
import Player from "./Player";
import RatingForm from "./RatingForm";
import ScriptView from "./ScriptView";

const RUNNING = new Set(["analysing", "planning", "writing", "queued", "synthesizing", "rewriting", "modulating", "mixing"]);

function saveJson(name: string, data: unknown) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  a.download = name + ".json";
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

  useEffect(() => { getHealth().then((h) => setServerMock(h.mock)).catch(fail); }, []);
  useEffect(() => { if (id) refresh(id); }, [id, refresh]);
  const stage = perf?.status.stage;
  useEffect(() => {
    if (!id || !stage || !RUNNING.has(stage)) return;
    const timer = window.setInterval(() => refresh(id), 1500);
    return () => window.clearInterval(timer);
  }, [id, stage, refresh]);

  async function onGenerate(req: GenerateRequest) {
    setError(null); setPerf(null); setActiveId(null);
    try {
      const result = await createPerformance(req);
      location.hash = result.id;
      setId(result.id);
    } catch (e) { fail(e); }
  }

  const running = stage ? RUNNING.has(stage) : false;
  const complete = perf?.status.stage === "done" && !!perf.timeline;
  const progress = perf?.status.progress ?? 0;

  return (
    <div className="site-shell">
      <header className="topbar">
        <a className="brand" href="#" onClick={() => { setId(null); setPerf(null); }}>
          <span className="brand-mark">ॐ</span><span>अनुभूति<span className="brand-sub">THE HARIKATHA STUDIO</span></span>
        </a>
        <nav><a href="#studio">Create</a><a href="#experience">Experience</a><a href="#script">Script</a></nav>
        <span className="open-source"><i /> OPEN SOURCE · LOCAL AI</span>
      </header>

      <section className="hero">
        <div className="hero-orbit orbit-one" /><div className="hero-orbit orbit-two" />
        <div className="hero-copy">
          <span className="eyebrow"><span /> A NEW WAY TO EXPERIENCE DEVOTION</span>
          <h1>Stories that<br /><em>touch the soul.</em></h1>
          <p>Ancient wisdom. Living stories. An immersive Harikatha experience, shaped by expressive AI narration and the timeless language of devotion.</p>
          <a className="hero-cta" href="#studio">Begin your journey <span>↗</span></a>
        </div>
        <div className="hero-art" aria-label="Abstract golden devotional mandala artwork">
          <div className="mandala mandala-a" /><div className="mandala mandala-b" />
          <div className="hero-symbol">श्री</div><div className="art-caption">श्रवणम् · मननम् · अनुभूतिः</div>
        </div>
        <div className="hero-foot"><span>01 / THE ART OF STORYTELLING</span><span>MADE WITH श्रद्धा</span></div>
      </section>

      <main className="workspace">
        <section className="section-heading" id="studio">
          <div><span className="eyebrow">YOUR STORY, REIMAGINED</span><h2>The story studio<span>.</span></h2></div>
          <p>Bring a story, a passage, or a moment of inspiration. We shape it into a traditional, emotionally rich narration.</p>
        </section>
        <GenerateForm busy={running} onSubmit={onGenerate} />
        {error && <p className="error">{error}</p>}
        {serverMock && <p className="notice">Demo mode is active. Switch MOCK_MODE=false after setting up Ollama and Indic Parler-TTS for real narration.</p>}

        {perf && (
          <section className="result-area">
            <div className="result-top">
              <div><span className="eyebrow">YOUR CREATION</span><h2>{perf.request.title}</h2></div>
              <span className={"status-pill " + (running ? "working" : complete ? "ready" : "idle")}>{running ? "IN THE MAKING" : complete ? "READY TO EXPERIENCE" : perf.status.stage.toUpperCase()}</span>
            </div>
            <p className="status-detail">{perf.status.detail || "Your Harikatha is taking shape."}</p>
            {running && <div className="progress-track"><span style={{ width: Math.round(progress * 100) + "%" }} /></div>}
            {perf.mock && <p className="notice">This performance was generated in mock mode. The audio is a test tone, not spoken narration.</p>}
            {perf.status.stage === "error" && <p className="error">{perf.status.detail}</p>}
            {perf.script && !complete && <button className="primary-button" disabled={running} onClick={() => render(perf.id).then(() => refresh(perf.id)).catch(fail)}>Render the experience <span>↗</span></button>}
            {complete && perf.timeline && (
              <div id="experience">
                <Player id={perf.id} files={perf.status.files ?? []} timeline={perf.timeline}
                  scriptSegments={perf.script?.segments ?? []} credits={perf.credits}
                  onActive={setActiveId} onDownloadJson={(name) => saveJson(name, name === "script" ? perf.script : perf.timeline)} />
                <RatingForm id={perf.id} />
              </div>
            )}
            {perf.script && <div id="script" className="script-area"><div className="section-heading compact"><div><span className="eyebrow">THE WORDS BEHIND THE WONDER</span><h2>Your Harikatha<span>.</span></h2></div></div>
              <ScriptView segments={perf.script.segments} activeId={activeId} editable={!running}
                onSave={(sid, text) => editSegment(perf.id, sid, text).then(() => refresh(perf.id)).catch(fail)} />
            </div>}
          </section>
        )}
      </main>
      <footer className="footer"><span className="footer-mark">ॐ</span><span>अनुभूति — Where stories become experience.</span><span>BUILT WITH DEVOTION & OPEN SOURCE</span></footer>
    </div>
  );
}
