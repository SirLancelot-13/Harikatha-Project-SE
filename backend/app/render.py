"""Render job, stages 6-9 of WORKFLOW_AND_STORAGE.md s2.2.

Modulation is a placeholder (gain only); pitch snapping to a raga comes later.
Mixing is in mixer.py: voice, drone, scene, cymbal pulse and bells, mastered to -16 LUFS.
"""
import shutil
import time
from array import array
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import audio, calibration, mixer, planner, store, tts
from .config import settings

MAX_FIT_ROUNDS = 2


def _synth_all(script: dict) -> tuple[dict[str, array], int]:
    segs = script["segments"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda seg: tts.synthesize(seg["text"], script["language"], seg["delivery"]["pace"], seg["delivery"].get("emotion", "devotional"), seg["delivery"].get("emphasis_words", [])), segs
        ))
    clips = {seg["id"]: samples for seg, (samples, _) in zip(segs, results, strict=True)}
    return clips, sum(hit for _, hit in results)


def _overruns(script: dict, clips: dict[str, array]) -> list[dict]:
    rate = settings.tts_sample_rate
    return [
        {"id": s["id"], "budget_sec": s["target_sec"], "actual_sec": audio.duration(clips[s["id"]], rate),
         "overrun": audio.duration(clips[s["id"]], rate) / s["target_sec"] - 1}
        for s in script["segments"]
    ]


def run_render_job(pid: str) -> None:
    timings: dict[str, float] = {}
    t0 = time.monotonic()

    def lap(name: str) -> None:
        nonlocal t0
        timings[name] = round(timings.get(name, 0) + time.monotonic() - t0, 2)
        t0 = time.monotonic()

    try:
        d = settings.performances_dir / pid
        script = store.read(pid, "script.json")
        rate = settings.tts_sample_rate
        cache_hits = 0

        store.write_status(pid, "synthesizing", progress=0.1)
        clips, hits = _synth_all(script)
        cache_hits += hits
        lap("tts")
        # Learn real speaking speed from the first pass (before any rewrite changes the text).
        calibration.update(
            script["language"], sum(len(s["text"]) for s in script["segments"]),
            sum(audio.duration(c, rate) for c in clips.values()),
        )
        cps = calibration.get(script["language"])

        for rnd in range(1, MAX_FIT_ROUNDS + 1):
            rows = _overruns(script, clips)
            late = [r for r in rows if r["overrun"] > settings.overrun_threshold]
            if not late:
                break
            store.write_status(pid, "rewriting", f"round {rnd}: {len(late)} segments", progress=0.35)
            by_id = {s["id"]: s for s in script["segments"]}
            for r in late:
                seg = by_id[r["id"]]
                limit = round(seg["target_sec"] * cps * 0.95)  # aim just under budget
                seg["text"] = planner.rewrite_segment(
                    pid, seg, f"Shorten to fit the time budget ({seg['target_sec']:.0f}s of speech).",
                    limit, script["language"],
                )
            v = store.save_script(
                pid, script, f"fit round {rnd}: shortened {', '.join(r['id'] for r in late)}"
            )
            store.write_stage(pid, f"07_fit_round{rnd}.json", {
                "script_version": v, "segments": rows, "shortened": [r["id"] for r in late],
            })
            clips, hits = _synth_all(script)
            cache_hits += hits
            lap("fit")

        store.write_status(pid, "modulating", progress=0.6)
        report, voiced = [], {}
        for seg in script["segments"]:
            gain = seg["delivery"]["gain_db"]
            voiced[seg["id"]] = audio.apply_gain(clips[seg["id"]], gain)
            audio.write_wav(d / "clips" / f"{seg['id']}.wav", voiced[seg["id"]], rate)
            report.append({"id": seg["id"], "gain_db": gain, "peak": round(audio.peak(voiced[seg["id"]]), 3),
                           "raga": None, "unsnapped_fraction": None})
        store.write_stage(pid, "08_modulation.json", {"note": "placeholder: gain only", "segments": report})
        lap("modulate")

        store.write_status(pid, "mixing", progress=0.8)
        floats = {k: np.asarray(v, dtype=np.float32) / 32768 for k, v in voiced.items()}
        result = mixer.build(script, floats, rate)
        if shutil.which("ffmpeg"):
            loudness = mixer.master(result.samples, rate, d)
        else:  # no ffmpeg: peak-normalised WAV only, no loudness target, no MP3
            peak = float(np.abs(result.samples).max()) or 1.0
            pcm = array("h", np.round(result.samples / peak * 0.89 * 32767).astype(np.int16).tolist())
            audio.write_wav(d / "performance.wav", pcm, rate)
            loudness = {"files": ["performance.wav"]}
        files = loudness["files"]
        total_sec = round(len(result.samples) / rate, 3)
        store.write_stage(pid, "09_mix.json", {**result.recipe, "loudness": loudness})
        store.write_json(pid, "credits.json", result.credits)
        lap("mix")

        rows = {r["id"]: r for r in _overruns(script, clips)}
        by_id = {s["id"]: s for s in script["segments"]}
        timeline = [{
            **t, "section": by_id[t["id"]]["section"], "type": by_id[t["id"]]["type"],
            "audio": f"clips/{t['id']}.wav", "budget_sec": by_id[t["id"]]["target_sec"],
            "overrun": round(rows[t["id"]]["overrun"], 3), "scene": by_id[t["id"]].get("scene"),\n            "emotion": by_id[t["id"]]["delivery"].get("emotion", "devotional"),
        } for t in result.timeline]
        version = store.read(pid, "meta.json")["script_versions"][-1]["v"]
        store.write_json(pid, "timeline.json", {
            "performance_id": pid, "sample_rate": rate, "voice_sa_hz": result.recipe["voice_f0_hz"],
            "target_sec": script["target_sec"], "total_sec": total_sec,
            "script_version": version, "segments": timeline,
        })
        meta = store.read(pid, "meta.json")
        store.update_meta(
            pid, timings_sec={**meta["timings_sec"], **timings},
            tts_chars=sum(len(s["text"]) for s in script["segments"]), tts_cache_hits=cache_hits,
        )
        store.write_status(pid, "done", progress=1.0, files=files, total_sec=total_sec)
    except Exception as e:  # noqa: BLE001 - surfaced to the UI via status.json
        store.write_status(pid, "error", f"{type(e).__name__}: {e}")
