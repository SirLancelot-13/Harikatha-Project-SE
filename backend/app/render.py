"""Render job, stages 6-9 of WORKFLOW_AND_STORAGE.md s2.2.

Modulation is a placeholder (gain only); pitch snapping to a raga comes later.
Mixing is in mixer.py: voice, drone, scene, cymbal pulse and bells, mastered to -16 LUFS.
"""
import shutil
import time
from array import array
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import audio, calibration, local, mixer, pitch, planner, store, tts
from .config import settings

MAX_FIT_ROUNDS = 2


def _synth_all(script: dict, backend: str) -> tuple[dict[str, array], int]:
    segs = script["segments"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda seg: tts.synthesize(
                seg["text"], script["language"], seg["delivery"]["pace"], seg["type"], backend
            ),
            segs,
        ))
    clips = {seg["id"]: samples for seg, (samples, _) in zip(segs, results, strict=True)}
    return clips, sum(hit for _, hit in results)


def _synth_with_fallback(
    script: dict, backend: str, reason: str | None
) -> tuple[dict[str, array], int, str, str | None]:
    """Synthesise every segment with ONE backend. If the local server fails part-way and the
    provider is "auto", redo the WHOLE script with Sarvam so the story keeps a single voice."""
    try:
        clips, hits = _synth_all(script, backend)
    except local.LocalError as e:
        if backend != "local" or settings.tts_provider != "auto":
            raise
        backend, reason = "sarvam", f"local TTS failed during rendering, used Sarvam instead: {e}"
        clips, hits = _synth_all(script, backend)
    return clips, hits, backend, reason


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
        backend, fallback = tts.pick_backend(script["language"])
        clips, hits, backend, fallback = _synth_with_fallback(script, backend, fallback)
        cache_hits += hits
        lap("tts")
        # Learn real speaking speed from the first pass (before any rewrite changes the text).
        voice = tts.voice_id(backend)
        calibration.update(
            script["language"], sum(len(s["text"]) for s in script["segments"]),
            sum(audio.duration(c, rate) for c in clips.values()), voice,
        )
        cps = calibration.get(script["language"], voice)

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
            previous = backend
            clips, hits, backend, fallback = _synth_with_fallback(script, backend, fallback)
            if backend != previous:  # fell back during the fit round: re-learn speed for this voice
                voice = tts.voice_id(backend)
                cps = calibration.get(script["language"], voice)
            cache_hits += hits
            lap("fit")

        store.write_status(pid, "modulating", progress=0.6)
        report, voiced, baseline = [], {}, {}
        for seg in script["segments"]:
            delivery = seg["delivery"]
            samples = audio.apply_gain(clips[seg["id"]], delivery["gain_db"])
            baseline[seg["id"]] = np.asarray(samples, dtype=np.float32) / 32768
            row = {"id": seg["id"], "gain_db": delivery["gain_db"], "raga": None, "unsnapped_fraction": None}
            lift = delivery.get("pitch_lift_st", 0.0)
            if lift > 0 and settings.pitch_modulation:
                shifted, row["pitch"] = pitch.lift(baseline[seg["id"]], rate, lift)
                if row["pitch"]["applied"]:
                    samples = array("h", np.round(shifted * 32767).astype(np.int16).tolist())
            voiced[seg["id"]] = samples
            row["peak"] = round(audio.peak(samples), 3)
            audio.write_wav(d / "clips" / f"{seg['id']}.wav", samples, rate)
            report.append(row)
        store.write_stage(pid, "08_modulation.json", {
            "note": "gain + gentle WORLD pitch lift on shlokas; no raga snapping yet", "segments": report,
        })
        lap("modulate")

        store.write_status(pid, "mixing", progress=0.8)
        floats = {k: np.asarray(v, dtype=np.float32) / 32768 for k, v in voiced.items()}
        result = mixer.build(script, floats, rate, pitch_ref=baseline)
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
            "overrun": round(rows[t["id"]]["overrun"], 3),
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
            tts_backend=backend, tts_fallback=fallback,
        )
        store.write_status(pid, "done", progress=1.0, files=files, total_sec=total_sec, tts_backend=backend)
    except Exception as e:  # noqa: BLE001 - surfaced to the UI via status.json
        store.write_status(pid, "error", f"{type(e).__name__}: {e}")
