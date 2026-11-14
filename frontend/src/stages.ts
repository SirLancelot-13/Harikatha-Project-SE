// The two phases of a performance and what each stage means, in plain words.
export const SCRIPT_STEPS = ["analysing", "planning", "writing", "scripted"] as const;
export const RENDER_STEPS = ["queued", "synthesizing", "rewriting", "modulating", "mixing", "done"] as const;

export const RUNNING = new Set([
  "analysing",
  "planning",
  "writing",
  "queued",
  "synthesizing",
  "rewriting",
  "modulating",
  "mixing",
]);

export const STAGE_HINT: Record<string, string> = {
  analysing: "Reading the story: who is in it, what happens, and its mood.",
  planning: "Dividing the time between the parts of the performance.",
  writing: "Writing each part in the language you chose, then reviewing it.",
  scripted: "The script is ready. Edit any line you like, then render it.",
  queued: "Waiting for its turn.",
  synthesizing: "Speaking the script, line by line.",
  rewriting: "Shortening lines that ran longer than their time.",
  modulating: "Shaping the voice, with a gentle lift on the shlokas.",
  mixing: "Adding the tanpura, bells and ambience under the voice.",
  done: "Ready to play.",
  error: "Something went wrong.",
};
