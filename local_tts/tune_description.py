"""Compare TTS descriptions objectively: pitch spread and loudness variation vs the Bulbul reference.

Run from backend/ (needs numpy + pyworld) with the local server up:  python ../local_tts/tune_description.py
A flat voice has a narrow pitch range and little loudness variation. This cannot judge naturalness or
intelligibility: always finish by listening to local_tts/samples/tune/.
"""
import io
import pathlib
import sys
import wave

import httpx
import numpy as np
import pyworld as pw

OUT = pathlib.Path(__file__).parent / "samples" / "tune"
LINES = {
    "calm": "सज्जनों, आज हम गजेन्द्र मोक्ष की कथा सुनेंगे। यह कथा भक्ति और पूर्ण समर्पण की कथा है।",
    "drama": "तभी एक मगरमच्छ ने उसका पैर पकड़ लिया! गजेन्द्र पूरी शक्ति से छूटने का प्रयास करने लगा, पर वह छूट न सका।",
    "devout": "तब गजेन्द्र ने कमल का फूल तोड़ा और पूरे हृदय से पुकारा, हे नारायण, मेरी रक्षा करो!",
}
TAIL = " The recording is of very high quality, with the speaker's voice sounding clear and very close up."
BASE = "{speaker} speaks at a moderate speed with a slightly expressive and animated delivery and a balanced pitch." + TAIL
LIVELY = "{speaker} speaks at a fairly fast, lively pace with an expressive, animated delivery and a balanced pitch." + TAIL
EMOTE = ("{speaker} speaks at a fairly fast, lively pace in an expressive and emotive voice, stressing the important "
         "words for emphasis, with a warm, devotional feeling and natural rises and falls in pitch." + TAIL)
# name -> (description, sampling temperature or None for the model default)
VARIANTS = {
    "OLD_baseline": (BASE, None),
    "NEW_default": (None, None),  # the server's built-in describe() for kind=narration
}


def metrics(x: np.ndarray, rate: int, chars: int) -> dict:
    f0, _ = pw.harvest(x.astype(np.float64), rate, f0_floor=70.0, f0_ceil=450.0, frame_period=5.0)
    voiced = f0 > 0
    v = f0[voiced]
    st = 12 * np.log2(v / np.median(v))
    both = voiced[1:] & voiced[:-1]  # consecutive voiced frames only
    jump = np.abs(12 * np.log2(f0[1:][both] / f0[:-1][both]))
    hop = int(0.025 * rate)
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt((frames**2).mean(1)) + 1e-9)
    speech = db > db.max() - 35
    return {"f0": float(np.median(v)), "std_st": float(st.std()), "range_st": float(np.percentile(st, 95) - np.percentile(st, 5)),
            "db_std": float(db[speech].std()), "cps": chars / (len(x) / rate),
            "jumpy": float((jump > 1.5).mean() * 100),  # % of 5 ms steps with a pitch jump over 1.5 semitones: wobble/instability
            "pause": float(1 - speech.mean()) * 100}  # % of time that is near-silent: pauses between phrases


def tighten(x: np.ndarray, rate: int, max_pause: float = 0.15, thresh_db: float = -38.0) -> np.ndarray:
    """PROTOTYPE: shorten internal silences longer than `max_pause` seconds (never touches the speech)."""
    win = int(0.01 * rate)
    n = len(x) // win
    e = 20 * np.log10(np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(1)) + 1e-9)
    quiet = e < e.max() + thresh_db
    out, i = [], 0
    while i < n:
        j = i
        while j < n and quiet[j] == quiet[i]:
            j += 1
        seg = x[i * win: j * win]
        internal = quiet[i] and i > 0 and j < n
        keep = int(max_pause * rate)
        out.append(np.concatenate([seg[: keep // 2], seg[-keep // 2:]]) if internal and len(seg) > keep else seg)
        i = j
    return np.concatenate([*out, x[n * win:]])


def _wav(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(np.round(np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def read(b: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(b)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768


def local(c: httpx.Client, text: str, desc: str | None, seed: int, temp: float | None) -> bytes:
    r = c.post("http://127.0.0.1:8100/synthesize", json={"text": text, "language_code": "hi-IN", "description": desc, "seed": seed, "temperature": temp, "kind": "narration"})
    r.raise_for_status()
    return r.content


def main() -> None:
    sys.path.insert(0, str(pathlib.Path.cwd()))  # run from backend/ so `app` imports

    rows: dict[str, list[dict]] = {}
    with httpx.Client(timeout=300) as c:
        for name, (desc, temp) in VARIANTS.items():
            for seed in (7, 21):
                for line, text in LINES.items():
                    wav = local(c, text, desc, seed, temp)
                    (OUT / f"{name}_{line}_s{seed}.wav").write_bytes(wav)
                    rows.setdefault(name, []).append(metrics(read(wav), 24000, len(text)))
    for line, text in LINES.items():  # offline prototype: the NEW default with long pauses shortened
        for seed in (7, 21):
            src = OUT / f"NEW_default_{line}_s{seed}.wav"
            if src.exists():
                y = tighten(read(src.read_bytes()), 24000)
                (OUT / f"NEW_tight_{line}_s{seed}.wav").write_bytes(
                    _wav(y)
                )
                rows.setdefault("NEW_tight_pauses", []).append(metrics(y, 24000, len(text)))
    for line, text in LINES.items():  # Bulbul reference: reuse the clips saved by an earlier run (no API calls)
        ref = OUT / f"BULBUL_{line}.wav"
        if ref.exists():
            rows.setdefault("BULBUL_reference", []).append(metrics(read(ref.read_bytes()), 24000, len(text)))
    print(f"{'variant':<20}{'f0':>5}{'p-std':>7}{'range':>7}{'loud':>6}{'c/s':>6}{'jumpy%':>8}{'pause%':>8}")
    for name, rs in rows.items():
        m = {k: np.mean([r[k] for r in rs]) for k in rs[0]}
        print(f"{name:<20}{m['f0']:>5.0f}{m['std_st']:>7.2f}{m['range_st']:>7.1f}{m['db_std']:>6.1f}{m['cps']:>6.1f}{m['jumpy']:>8.1f}{m['pause']:>8.1f}")


if __name__ == "__main__":
    sys.exit(main())
