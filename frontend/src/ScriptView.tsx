import { useLayoutEffect, useRef, useState } from "react";
import type { Segment } from "./api";

const SECTION_LABEL = (s: string) => {
  if (s.startsWith("nirupana")) return `Story part ${s.split("_")[1]}`;
  if (s === "shloka_meaning") return "Shloka and its meaning";
  const text = s.replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
};

type Edit = { text?: string; pitch_lift_st?: number };

/** A textarea that grows to fit its text, so a whole song or paragraph is visible while editing. */
function AutoTextarea({ value, onChange, disabled, label }: { value: string; onChange: (v: string) => void; disabled: boolean; label: string }) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const fit = () => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  };
  useLayoutEffect(() => {
    fit();
    void document.fonts?.ready.then(fit); // the web font changes line heights once it arrives
  }, [value]);
  return <textarea ref={ref} className="script-text" aria-label={label} value={value} rows={1} disabled={disabled} onChange={(e) => onChange(e.target.value)} />;
}

function SegmentRow({ seg, active, editable, onSave }: { seg: Segment; active: boolean; editable: boolean; onSave: (id: string, edit: Edit) => Promise<void> }) {
  const [text, setText] = useState(seg.text);
  const [lift, setLift] = useState(seg.delivery.pitch_lift_st);
  const dirty = text !== seg.text || lift !== seg.delivery.pitch_lift_st;

  return (
    <div className={`segment type-${seg.type}${active ? " active" : ""}`}>
      <div className="meta">
        <span className={`tag tag-type type-${seg.type}`}>{seg.type}</span>
        <span className={`tag tag-origin ${seg.origin}`}>{seg.origin}</span>
        <span className="secs">{seg.target_sec.toFixed(0)}s</span>
        {seg.source_ids.length > 0 && <span className="source">source: {seg.source_ids.join(", ")}</span>}
      </div>
      <AutoTextarea value={text} onChange={setText} disabled={!editable} label={`Text of ${seg.id}`} />
      <div className="segment-foot">
        <label className="lift">
          <span>Pitch lift (semitones, 0-3)</span>
          <input
            type="number"
            min={0}
            max={3}
            step={0.5}
            value={lift}
            disabled={!editable}
            onChange={(e) => setLift(Math.min(3, Math.max(0, +e.target.value || 0)))}
          />
        </label>
        {dirty && editable && (
          <span className="unsaved">
            Unsaved changes
            <button
              className="btn primary small"
              onClick={() =>
                onSave(seg.id, {
                  text: text !== seg.text ? text : undefined,
                  pitch_lift_st: lift !== seg.delivery.pitch_lift_st ? lift : undefined,
                })
              }
            >
              Save edit
            </button>
          </span>
        )}
      </div>
    </div>
  );
}

export default function ScriptView({
  segments,
  activeId,
  editable,
  onSave,
}: {
  segments: Segment[];
  activeId: string | null;
  editable: boolean;
  onSave: (id: string, edit: Edit) => Promise<void>;
}) {
  const sections = [...new Set(segments.map((s) => s.section))];
  return (
    <div className="script">
      <div className="script-head">
        <h2 className="section-heading">The script</h2>
        <p className="hint">Edit any line or its pitch lift, then save. Every save keeps the earlier version.</p>
      </div>
      {sections.map((sec) => {
        const inSection = segments.filter((s) => s.section === sec);
        const seconds = inSection.reduce((sum, s) => sum + s.target_sec, 0);
        return (
          <section key={sec} className="card script-section">
            <header className="section-head">
              <h3>{SECTION_LABEL(sec)}</h3>
              <span className="section-secs">about {seconds.toFixed(0)}s</span>
            </header>
            {inSection.map((s) => (
              // key includes text so a saved or regenerated segment resets its draft
              <SegmentRow key={`${s.id}:${s.text}:${s.delivery.pitch_lift_st}`} seg={s} active={s.id === activeId} editable={editable} onSave={onSave} />
            ))}
          </section>
        );
      })}
    </div>
  );
}
