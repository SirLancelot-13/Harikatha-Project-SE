export type Status = {
  stage: string;
  detail: string;
  progress: number | null;
  script_version: number | null;
  updated_at: string;
  files?: string[];
  total_sec?: number;
};

export type Segment = {
  id: string;
  section: string;
  type: "shloka" | "narration" | "song";
  text: string;
  origin: "original" | "traditional";
  source_ids: string[];
  target_sec: number;
  delivery?: { emotion?: string; emphasis_words?: string[]; pace?: number };
  scene?: string | null;
};

export type Script = { title: string; language: string; target_sec: number; segments: Segment[] };

export type TimelineSegment = { id: string; start: number; end: number; overrun: number; section?: string; type?: string; scene?: string | null; emotion?: string };
export type Timeline = { total_sec: number; script_version: number; segments: TimelineSegment[] };

export type Credit = {
  file: string;
  title: string | null;
  author: string | null;
  source: string | null;
  licence: string | null;
  requires_attribution: boolean;
};

export type Performance = {
  id: string;
  request: { title: string; story: string; language: string; duration_minutes: number };
  script: Script | null;
  timeline: Timeline | null;
  credits: Credit[];
  mock: boolean;
  status: Status;
};

export type GenerateRequest = Performance["request"];

export type Rating = {
  rater_id: string;
  clarity: number;
  musicality: number;
  harikatha_feel: number;
  fidelity: number;
  comment: string;
};

const BASE = "/api/v1";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${(await res.text()).slice(0, 200)}`);
  return res.json();
}

const send = (method: string, url: string, body?: unknown) =>
  fetch(`${BASE}${url}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const createPerformance = (req: GenerateRequest) =>
  send("POST", "/performances", req).then((r) => json<{ id: string; status: Status }>(r));

export const getPerformance = (id: string) =>
  fetch(`${BASE}/performances/${id}`).then((r) => json<Performance>(r));

export const getHealth = () => fetch(`${BASE}/health`).then((r) => json<{ ok: boolean; mock: boolean }>(r));

export const render = (id: string) => send("POST", `/performances/${id}/render`).then((r) => json<Status>(r));

export const editSegment = (id: string, sid: string, text: string) =>
  send("PATCH", `/performances/${id}/segments/${sid}`, { text }).then((r) =>
    json<{ script_version: number; segment: Segment }>(r),
  );

export const rate = (id: string, rating: Rating) =>
  send("POST", `/performances/${id}/ratings`, rating).then((r) => json<{ ok: boolean }>(r));

export const audioUrl = (id: string, fmt: "mp3" | "wav") => `${BASE}/performances/${id}/audio?fmt=${fmt}`;
