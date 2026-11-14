import { useState } from "react";
import type { GenerateRequest } from "./api";

const LANGUAGES = [
  ["en-IN", "English"],
  ["hi-IN", "Hindi"],
  ["te-IN", "Telugu"],
  ["ta-IN", "Tamil"],
  ["kn-IN", "Kannada"],
];

export default function GenerateForm({ busy, onSubmit }: { busy: boolean; onSubmit: (r: GenerateRequest) => void }) {
  const [title, setTitle] = useState("");
  const [story, setStory] = useState("");
  const [language, setLanguage] = useState("en-IN");
  const [minutes, setMinutes] = useState(4);

  return (
    <form
      className="card form"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit({ title, story, language, duration_minutes: minutes });
      }}
    >
      <h2 className="card-title">Tell me a story</h2>

      <label className="field">
        <span className="label">Title</span>
        <input placeholder="e.g. Gajendra Moksha" value={title} onChange={(e) => setTitle(e.target.value)} required />
      </label>

      <label className="field">
        <span className="label">Story</span>
        <textarea
          placeholder="Write or paste the story here, as you would tell it to a friend"
          rows={6}
          value={story}
          onChange={(e) => setStory(e.target.value)}
          required
        />
      </label>

      <div className="form-row">
        <label className="field">
          <span className="label">Language</span>
          <select value={language} onChange={(e) => setLanguage(e.target.value)}>
            {LANGUAGES.map(([v, l]) => (
              <option key={v} value={v}>
                {l}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="label">Minutes</span>
          <input type="number" min={2} max={60} value={minutes} onChange={(e) => setMinutes(+e.target.value)} />
        </label>
        <button type="submit" className="btn primary" disabled={busy}>
          Generate script
        </button>
      </div>

      <p className="hint">
        The story can be in any language. The language you choose is the one it will be told in. A performance runs at least 2 minutes.
      </p>
    </form>
  );
}
