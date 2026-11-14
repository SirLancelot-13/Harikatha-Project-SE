import { useRef, useState } from "react";
import { audioUrl, type Credit, type Timeline } from "./api";

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export default function Player({
  id,
  files,
  timeline,
  credits,
  onActive,
  onDownloadJson,
}: {
  id: string;
  files: string[];
  timeline: Timeline;
  credits: Credit[];
  onActive: (segId: string | null) => void;
  onDownloadJson: (name: "script" | "timeline") => void;
}) {
  const fmt = files.includes("performance.mp3") ? "mp3" : "wav";
  const [t, setT] = useState(0);
  const audio = useRef<HTMLAudioElement>(null);

  const seek = (to: number) => {
    if (!audio.current) return;
    audio.current.currentTime = to;
    void audio.current.play();
  };

  return (
    <section className="card player">
      <h2 className="card-title">Performance</h2>
      <audio
        ref={audio}
        controls
        src={audioUrl(id, fmt)}
        onTimeUpdate={(e) => {
          const now = e.currentTarget.currentTime;
          setT(now);
          onActive(timeline.segments.find((s) => now >= s.start && now < s.end + 0.35)?.id ?? null);
        }}
        onEnded={() => onActive(null)}
      />

      <div className="timeline-wrap">
        <div className="timeline" role="group" aria-label="Performance timeline: click a part to jump to it" title={`${t.toFixed(1)}s / ${timeline.total_sec}s`}>
          {timeline.segments.map((s) => (
            <button
              key={s.id}
              type="button"
              className={`tick type-${s.type ?? "narration"}${t >= s.start && t < s.end + 0.35 ? " on" : ""}`}
              style={{ left: `${(s.start / timeline.total_sec) * 100}%`, width: `${((s.end - s.start) / timeline.total_sec) * 100}%` }}
              title={`${s.id}${s.type ? ` · ${s.type}` : ""}`}
              aria-label={`Jump to ${s.id}`}
              onClick={() => seek(s.start)}
            />
          ))}
          <span className="playhead" style={{ left: `${Math.min(100, (t / timeline.total_sec) * 100)}%` }} aria-hidden="true" />
        </div>
        <div className="timeline-foot">
          <span className="clock">
            {clock(t)} / {clock(timeline.total_sec)}
          </span>
          <span className="legend" aria-hidden="true">
            <i className="swatch type-shloka" /> shloka <i className="swatch type-song" /> song <i className="swatch type-narration" /> narration
          </span>
        </div>
      </div>

      <div className="row downloads">
        {files.map((f) => (
          <a key={f} className="chip" href={audioUrl(id, f.endsWith("mp3") ? "mp3" : "wav")} download>
            {f}
          </a>
        ))}
        <button className="chip" onClick={() => onDownloadJson("script")}>
          script.json
        </button>
        <button className="chip" onClick={() => onDownloadJson("timeline")}>
          timeline.json
        </button>
      </div>

      {credits.length > 0 && (
        <p className="credits">
          Sounds:{" "}
          {credits.map((c, i) => (
            <span key={c.file}>
              {i > 0 && "; "}
              <a href={c.source ?? undefined} target="_blank" rel="noreferrer">
                {c.title}
              </a>{" "}
              by {c.author} ({c.licence})
            </span>
          ))}
        </p>
      )}
    </section>
  );
}
