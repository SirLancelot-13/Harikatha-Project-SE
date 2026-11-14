import { useState } from "react";
import { rate } from "./api";

const FIELDS = ["clarity", "musicality", "harikatha_feel", "fidelity"] as const;

export default function RatingForm({ id }: { id: string }) {
  const [scores, setScores] = useState<Record<(typeof FIELDS)[number], number>>({
    clarity: 3,
    musicality: 3,
    harikatha_feel: 3,
    fidelity: 3,
  });
  const [comment, setComment] = useState("");
  const [sent, setSent] = useState(false);

  if (sent) return <p className="card thanks">Thanks for rating.</p>;
  return (
    <form
      className="card form"
      onSubmit={async (e) => {
        e.preventDefault();
        // Pseudonymous rater id: never a name or email.
        await rate(id, { rater_id: crypto.randomUUID().slice(0, 8), ...scores, comment });
        setSent(true);
      }}
    >
      <h2 className="card-title">Rate this performance</h2>
      <p className="hint">1 is poor, 5 is excellent. Your rating helps us tune how it sounds.</p>
      <div className="ratings">
        {FIELDS.map((f) => (
          <label key={f} className="rating">
            <span className="rating-name">{f.replace("_", " ")}</span>
            <input
              type="range"
              min={1}
              max={5}
              value={scores[f]}
              aria-valuetext={`${scores[f]} out of 5`}
              onChange={(e) => setScores({ ...scores, [f]: +e.target.value })}
            />
            <span className="rating-value">{scores[f]}</span>
          </label>
        ))}
      </div>
      <textarea placeholder="Comment (optional)" aria-label="Comment (optional)" rows={3} value={comment} onChange={(e) => setComment(e.target.value)} />
      <button type="submit" className="btn primary">
        Submit rating
      </button>
    </form>
  );
}
