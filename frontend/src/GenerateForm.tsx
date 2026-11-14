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
      <input placeholder="Title" value={title} onChange={(e) => setTitle(e.target.value)} required />
      <textarea placeholder="Story" rows={6} value={story} onChange={(e) => setStory(e.target.value)} required />
      <div className="row">
        <select value={language} onChange={(e) => setLanguage(e.target.value)}>
          {LANGUAGES.map(([v, l]) => (
            <option key={v} value={v}>
              {l}
            </option>
          ))}
        </select>
        <label>
          Minutes{" "}
          <input type="number" min={2} max={60} value={minutes} onChange={(e) => setMinutes(+e.target.value)} />
        </label>
        <button type="submit" disabled={busy}>
          Generate script
        </button>
      </div>
    </form>
  );
}
