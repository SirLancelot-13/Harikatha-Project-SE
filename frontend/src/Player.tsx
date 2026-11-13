import { useMemo, useState } from "react";
import { audioUrl, type Credit, type Segment, type Timeline } from "./api";

export default function Player({ id, files, timeline, scriptSegments, credits, onActive, onDownloadJson }: {
  id: string; files: string[]; timeline: Timeline; scriptSegments: Segment[]; credits: Credit[];
  onActive: (segId: string | null) => void; onDownloadJson: (name: "script" | "timeline") => void;
}) {
  const fmt = files.includes("performance.mp3") ? "mp3" : "wav";
  const [time, setTime] = useState(0);
  const active = useMemo(() => {
    const current = timeline.segments.find((s) => time >= s.start && time < s.end + 0.35);
    return current ? scriptSegments.find((s) => s.id === current.id) ?? null : null;
  }, [time, timeline.segments, scriptSegments]);
  const currentTimeline = timeline.segments.find((s) => active && s.id === active.id);
  const sceneIndex = Math.max(0, timeline.segments.findIndex((s) => active && s.id === active.id));

  return (
    <section className="experience-card">
      <div className="experience-head"><div><span className="eyebrow">THE IMMERSIVE LISTENING ROOM</span><h3>Enter the story<span>.</span></h3></div><span className="live-label"><i /> LIVE EXPERIENCE</span></div>
      <div className={"scene-stage scene-" + (sceneIndex % 4)}>
        <div className="scene-grain" /><div className="scene-glow" />
        <div className="scene-topline"><span>{active?.section?.replaceAll("_", " ").toUpperCase() ?? "THE JOURNEY BEGINS"}</span><span>{currentTimeline?.emotion ?? "DEVOTIONAL NARRATION"}</span></div>
        <div className="scene-content">
          <span className="scene-kicker">॥ श्री हरिः ॥</span>
          <p className="live-script" aria-live="polite">{active?.text ?? "Let the story unfold. Press play and listen with your heart."}</p>
          {active?.delivery?.emphasis_words?.length ? <div className="emphasis-chips">{active.delivery.emphasis_words.map((w) => <span key={w}>{w}</span>)}</div> : null}
        </div>
        <div className="scene-bottom"><span>अनुभूतिः</span><span>{String(sceneIndex + 1).padStart(2, "0")} / {String(timeline.segments.length).padStart(2, "0")}</span></div>
      </div>
      <div className="audio-console">
        <audio controls preload="metadata" src={audioUrl(id, fmt)}
          onTimeUpdate={(e) => { const now = e.currentTarget.currentTime; setTime(now); onActive(timeline.segments.find((s) => now >= s.start && now < s.end + 0.35)?.id ?? null); }}
          onEnded={() => onActive(null)} />
        <div className="timeline" title={time.toFixed(1) + "s / " + timeline.total_sec + "s"}>
          {timeline.segments.map((seg) => <div key={seg.id} className={"tick" + (active?.id === seg.id ? " on" : "")}
            style={{ width: ((seg.end - seg.start) / timeline.total_sec * 100) + "%" }} />)}
        </div>
        <div className="row player-actions">
          {files.map((file) => <a key={file} href={audioUrl(id, file.endsWith("mp3") ? "mp3" : "wav")} download>↓ {file}</a>)}
          <button onClick={() => onDownloadJson("script")}>↓ Script JSON</button>
          <button onClick={() => onDownloadJson("timeline")}>↓ Timeline JSON</button>
        </div>
      </div>
      {credits.length > 0 && <p className="credits">Sound credits: {credits.map((c, i) => <span key={c.file}>{i > 0 ? "; " : ""}<a href={c.source ?? undefined} target="_blank" rel="noreferrer">{c.title}</a> by {c.author} ({c.licence})</span>)}</p>}
    </section>
  );
}
