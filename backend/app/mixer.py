"""Layered mixer. See WORKFLOW_AND_STORAGE.md stage 9.

Layers on one sample-accurate timeline: voice, tanpura drone, scene bed, cymbal pulse
under songs, and bells. Beds are levelled relative to the voice, ducked under speech
and faded in before the first word / out after the last, so there is no dead air.
"""
import json
import math
import re
import subprocess
from dataclasses import dataclass, field

import numpy as np

from . import sounds
from .config import settings

PRE_ROLL = 3.0  # drone and opening bell before the first word
POST_ROLL = 5.0  # closing bell rings out over a fading drone
PAUSE = {"narration": 0.15, "into_song": 0.40, "out_of_song": 0.35, "section": 0.80, "after_shloka": 0.50}
EDGE_FADE = 0.010  # per-clip fade to avoid clicks at joins
TRIM_DB = -45  # silence threshold relative to the clip's peak
TRIM_MARGIN = 0.02

# Bed levels in dB relative to the voice's RMS.
DRONE_OPEN, DRONE_SPEECH, DRONE_SONG = -30, -40, -40
SCENE_DB = -30
PULSE_DB, BELL_MAIN_DB, BELL_SOFT_DB = -20, -10, -20
TEMPO_BPM = 88
ACCENTS = [1.0, 0.35, 0.5, 0.35, 0.7, 0.35, 0.5, 0.35]  # 8-beat cycle, strongest on beat 1
ENV_RATE = 50  # Hz, control rate of the ducking envelopes
ATTACK, RELEASE = 0.15, 0.6
TARGET_LUFS, TARGET_TP = -16.0, -1.5


def db(x: float) -> float:
    return 10 ** (x / 20)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def trim_silence(x: np.ndarray, rate: int) -> np.ndarray:
    """Cut the silence TTS pads around a clip, keeping a short margin, and fade the edges."""
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak == 0:
        return x
    win = max(1, int(rate * 0.01))
    cs = np.concatenate([[0.0], np.cumsum(np.square(x, dtype=np.float64))])
    env = np.sqrt((cs[win:] - cs[:-win]) / win)
    loud = np.nonzero(env > peak * db(TRIM_DB))[0]
    if len(loud) == 0:
        return x
    m = int(TRIM_MARGIN * rate)
    out = x[max(0, loud[0] - m): min(len(x), loud[-1] + win + m)].copy()
    f = min(len(out) // 2, int(EDGE_FADE * rate))
    ramp = np.linspace(0, 1, f, dtype=np.float32)
    out[:f] *= ramp
    out[len(out) - f:] *= ramp[::-1]
    return out


def estimate_f0(x: np.ndarray, rate: int, fmin: int = 70, fmax: int = 400, max_sec: int = 90):
    """Median pitch of the voiced frames (autocorrelation); None if nothing is voiced."""
    x = x[: max_sec * rate]
    n = int(0.04 * rate)
    if len(x) < 4 * n:
        return None
    starts = np.arange(0, len(x) - n, n // 2)
    frames = np.stack([x[i: i + n] for i in starts]).astype(np.float64)
    frames -= frames.mean(axis=1, keepdims=True)
    energy = (frames**2).sum(axis=1)
    spec = np.fft.rfft(frames * np.hanning(n), 2 * n)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:, :n]
    ac /= ac[:, :1] + 1e-12
    lo, hi = int(rate / fmax), int(rate / fmin)
    lag = ac[:, lo:hi].argmax(axis=1) + lo
    strength = ac[np.arange(len(lag)), lag]
    voiced = (energy > 0.25 * energy.mean()) & (strength > 0.5)
    return float(np.median(rate / lag[voiced])) if voiced.sum() >= 10 else None


def semitone_shift(from_hz: float, to_hz: float) -> float:
    """Smallest shift (in semitones, within +/-6) taking pitch class `from_hz` to `to_hz`."""
    d = 12 * math.log2(to_hz / from_hz)
    return ((d + 6) % 12) - 6


def pitch_shift(x: np.ndarray, semitones: float) -> np.ndarray:
    """Resample shift (changes length, which is fine for a looped drone)."""
    r = 2 ** (semitones / 12)
    return np.interp(np.arange(0, len(x) - 1, r), np.arange(len(x)), x).astype(np.float32)


def make_loop(x: np.ndarray, rate: int, xfade: float = 3.0) -> np.ndarray:
    """Equal-power crossfade the tail into the head so the loop point is continuous."""
    n = min(int(xfade * rate), len(x) // 3)
    if n < 2:
        return x
    t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
    body = x[: len(x) - n].copy()
    body[:n] = x[:n] * np.sin(t) + x[len(x) - n:] * np.cos(t)
    return body


def tile_to(x: np.ndarray, n: int) -> np.ndarray:
    return np.tile(x, n // len(x) + 1)[:n]


def smooth(target: np.ndarray, attack: float = ATTACK, release: float = RELEASE) -> np.ndarray:
    """One-pole smoothing with a fast fall (duck) and a slow rise (swell back)."""
    ka, kr = 1 - math.exp(-1 / (attack * ENV_RATE)), 1 - math.exp(-1 / (release * ENV_RATE))
    out = np.empty_like(target)
    g = target[0]
    for i, t in enumerate(target):
        g += (t - g) * (ka if t < g else kr)
        out[i] = g
    return out


def _pause_before(prev: dict | None, cur: dict) -> float:
    if prev is None:
        return 0.0
    if cur["section"] != prev["section"]:
        if cur["section"].startswith("nirupana") and prev["section"].startswith("nirupana"):
            return PAUSE["narration"]  # consecutive story parts are one continuous telling
        return PAUSE["section"]
    if prev["type"] == "shloka" and cur["type"] == "narration":
        return PAUSE["after_shloka"]  # let the verse land before its meaning begins
    if cur["type"] == "song":
        return PAUSE["into_song"]
    if prev["type"] == "song":
        return PAUSE["out_of_song"]
    return PAUSE["narration"]


@dataclass
class Mix:
    samples: np.ndarray
    timeline: list[dict]  # id, start, end on the final timeline
    recipe: dict
    credits: list[dict] = field(default_factory=list)


def _credit(bed: sounds.Bed) -> dict | None:
    if bed.meta is None:
        return None
    m = bed.meta
    return {"file": bed.file, "title": m.get("title"), "author": m.get("author"),
            "source": m.get("source"), "licence": m.get("licence"),
            "requires_attribution": bool(m.get("requires_attribution"))}


def build(script: dict, clips: dict[str, np.ndarray], rate: int,
          pitch_ref: dict[str, np.ndarray] | None = None) -> Mix:
    """`pitch_ref`: unmodulated clips used to measure the voice's tonic, so an emphasis lift on
    some segments does not move the drone."""
    off = {x.strip() for x in settings.mix_disable.split(",") if x.strip()}  # MIX_DISABLE, for tuning by ear
    segs = script["segments"]
    trimmed = {s["id"]: trim_silence(clips[s["id"]], rate) for s in segs}

    # 1. voice placement: deliberate short pauses instead of fixed silence
    spans, cursor, prev = [], PRE_ROLL, None
    for s in segs:
        start = cursor + _pause_before(prev, s)
        end = start + len(trimmed[s["id"]]) / rate
        spans.append((s, start, end))
        cursor, prev = end, s
    total = cursor + POST_ROLL
    n = int(total * rate)
    voice = np.zeros(n, dtype=np.float32)
    for s, start, end in spans:
        i = int(start * rate)
        voice[i: i + len(trimmed[s["id"]])] += trimmed[s["id"]]
    voice_rms = rms(np.concatenate([trimmed[s["id"]] for s in segs])) or 1e-3

    # 2. drone: pick the recording needing the smallest shift to the voice's tonic
    ref = {s["id"]: trim_silence((pitch_ref or clips)[s["id"]], rate) for s in segs}
    f0 = estimate_f0(np.concatenate([ref[s["id"]] for s in segs]), rate)
    drones = sounds.tanpuras(rate)
    if f0:
        shifts = [semitone_shift(d.sa_hz, f0) for d in drones]
        k = min(range(len(drones)), key=lambda i: abs(shifts[i]))
        shift = shifts[k]
    else:
        k, shift = 0, 0.0
    drone = drones[k]
    loop = make_loop(pitch_shift(drone.samples, shift), rate)
    drone_bed = tile_to(loop, n) * (voice_rms * db(0) / (rms(loop) or 1))

    # 3. control envelopes at ENV_RATE: where speech / songs happen
    ne = int(total * ENV_RATE) + 1
    speech, song = np.zeros(ne), np.zeros(ne)
    for s, start, end in spans:
        a, b = int(max(0, start - 0.1) * ENV_RATE), int((end + 0.1) * ENV_RATE) + 1
        speech[a:b] = 1
        if s["type"] == "song":
            song[a:b] = 1
    target_db = np.where(song > 0, DRONE_SONG, np.where(speech > 0, DRONE_SPEECH, DRONE_OPEN))
    env_t = np.arange(ne) / ENV_RATE
    audio_t = np.arange(n) / rate
    drone_env = np.interp(audio_t, env_t, smooth(10 ** (target_db / 20)))
    drone_env *= np.clip(audio_t / (PRE_ROLL - 1.0), 0, 1)  # fade in before the first word
    drone_env *= np.clip((total - audio_t) / 4.0, 0, 1)  # fade out over the last 4 s
    mix = voice + (0 if "drone" in off else drone_bed * drone_env)

    # 4. scene bed over the nirupana sections, crossfaded in and out
    scene = sounds.scene(rate)
    nir = [(st, en) for s, st, en in spans if s["section"].startswith("nirupana")]
    scene = None if "scene" in off else scene
    if scene is not None and nir:
        bed = tile_to(make_loop(scene.samples, rate), n)
        bed = bed * (voice_rms * db(SCENE_DB) / (rms(bed) or 1))
        a, b = nir[0][0] - PAUSE["section"] / 2, nir[-1][1] + PAUSE["section"] / 2
        env = np.minimum(np.clip((audio_t - (a - 1.25)) / 2.5, 0, 1), np.clip(((b + 1.5) - audio_t) / 3.0, 0, 1))
        mix += bed * env.astype(np.float32)

    # 5. cymbal pulse under songs: soft accents on an 8-beat cycle, faded in and out
    cymbal = sounds.one_shot("cymbals", rate)
    hit = cymbal.samples[: int(1.5 * rate)].copy()
    hit[-int(0.3 * rate):] *= np.linspace(1, 0, int(0.3 * rate), dtype=np.float32)
    hit *= voice_rms * db(PULSE_DB) / (rms(hit[: rate]) or 1)
    beat = 60 / TEMPO_BPM
    for s, start, end in spans:
        if s["type"] != "song" or "pulse" in off:
            continue
        count = int((end - start) / beat)
        for j in range(count):
            ramp = min(1.0, (j + 1) / 2, (count - j) / 2)
            i = int((start + j * beat) * rate)
            seg = hit[: max(0, n - i)]
            mix[i: i + len(seg)] += seg * ACCENTS[j % 8] * ramp

    # 6. bells: open, a soft chime into each nirupana, close (tails ring over what follows)
    bell = sounds.one_shot("bell", rate)
    bell_s = bell.samples[: 8 * rate].copy()
    bell_s[-int(1.5 * rate):] *= np.linspace(1, 0, int(1.5 * rate), dtype=np.float32)
    bell_s *= voice_rms / (rms(bell_s[: 2 * rate]) or 1)
    events = [(0.4, BELL_MAIN_DB)]
    first_story = next((start for s, start, _ in spans if s["section"].startswith("nirupana")), None)
    if first_story is not None:  # one soft chime as the story begins
        events.append((first_story - PAUSE["section"] / 2, BELL_SOFT_DB))
    events.append((spans[-1][2] + 0.5, BELL_MAIN_DB))
    for t, level in [] if "bells" in off else events:
        i = max(0, int(t * rate))
        seg = bell_s[: max(0, n - i)]
        mix[i: i + len(seg)] += seg * db(level)

    has_song = any(s["type"] == "song" for s in segs)
    used = (([] if "drone" in off else [drone]) + ([scene] if scene is not None and nir else [])
            + ([] if "bells" in off else [bell]) + ([] if "pulse" in off or not has_song else [cymbal]))
    credits = [c for c in map(_credit, used) if c]
    recipe = {
        "pre_roll_sec": PRE_ROLL, "post_roll_sec": POST_ROLL, "pauses_sec": PAUSE,
        "voice_f0_hz": round(f0, 1) if f0 else None, "voice_rms_dbfs": round(20 * math.log10(voice_rms), 1),
        "drone": {"file": drone.file or "synthesised", "sa_hz": drone.sa_hz, "shift_semitones": round(shift, 2)},
        "scene": scene.file if scene is not None and nir else None,
        "bell": bell.file or "synthesised", "cymbals": cymbal.file or "synthesised",
        "tempo_bpm": TEMPO_BPM, "bells_at_sec": [[round(t, 2), lv] for t, lv in events],
        "levels_db_rel_voice": {"drone_open": DRONE_OPEN, "drone_speech": DRONE_SPEECH,
                                "drone_song": DRONE_SONG, "scene": SCENE_DB, "pulse": PULSE_DB},
        "voice_start_sec": round(spans[0][1], 3), "disabled_layers": sorted(off),
    }
    timeline = [{"id": s["id"], "start": round(st, 3), "end": round(en, 3)} for s, st, en in spans]
    return Mix(mix.astype(np.float32), timeline, recipe, credits)


# --- mastering -----------------------------------------------------------------

def _ffmpeg(args: list[str], data: bytes) -> str:
    return subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *args], input=data,
                          capture_output=True, check=True).stderr.decode(errors="replace")


def _json_block(text: str) -> dict:
    return json.loads(re.findall(r"\{[^{}]*\}", text)[-1])


def master(samples: np.ndarray, rate: int, out_dir, want_mp3: bool = True) -> dict:
    """Two-pass loudnorm to -16 LUFS / -1.5 dBTP; writes performance.wav (PCM16) and .mp3."""
    peak = float(np.abs(samples).max())
    pcm = (samples * (0.98 / peak if peak > 0.98 else 1.0)).astype("<f4").tobytes()
    src = ["-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "-"]
    ln = f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA=11"
    first = _json_block(_ffmpeg([*src, "-af", f"{ln}:print_format=json", "-f", "null", "-"], pcm))
    measured = (f"{ln}:measured_I={first['input_i']}:measured_TP={first['input_tp']}:"
                f"measured_LRA={first['input_lra']}:measured_thresh={first['input_thresh']}:"
                f"offset={first['target_offset']}:linear=true:print_format=json")
    outs = ["-map", "[a]", "-c:a", "pcm_s16le", str(out_dir / "performance.wav")]
    graph = f"[0:a]{measured},aresample={rate}"
    if want_mp3:
        graph += ",asplit=2[a][b]"
        outs += ["-map", "[b]", "-b:a", "192k", str(out_dir / "performance.mp3")]
    else:
        graph += "[a]"
    second = _json_block(_ffmpeg(["-y", *src, "-filter_complex", graph, *outs], pcm))
    return {"input_lufs": float(first["input_i"]), "output_lufs": float(second["output_i"]),
            "output_true_peak": float(second["output_tp"]),
            "files": ["performance.wav"] + (["performance.mp3"] if want_mp3 else [])}
