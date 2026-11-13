"""Gentle pitch lift for emphasis (used on shlokas), via the WORLD vocoder.

WORLD splits a clip into pitch (f0), spectral envelope and aperiodicity. We scale only
the pitch, so the voice keeps its own timbre and does not turn chipmunk-like, then
resynthesise. The lift eases in and out (raised cosine) so segment edges never jump.
Deliberately capped: this is an emphasis, not a change of register.
"""
import numpy as np
import pyworld as pw

MAX_LIFT_ST = 3.0  # hard cap in semitones: beyond this speech starts to sound unnatural
FRAME_MS = 5.0
RAMP_SEC = 0.25  # time to ease in and out
MIN_VOICED = 0.15  # skip clips that are mostly unvoiced (nothing to lift)


def contour(times: np.ndarray, peak_st: float) -> np.ndarray:
    """Semitones to add at each analysis frame: eased in, plateau, eased out."""
    total = float(times[-1]) if len(times) else 0.0
    ramp = max(1e-3, min(RAMP_SEC, total / 4))
    w = np.minimum(np.clip(times / ramp, 0, 1), np.clip((total - times) / ramp, 0, 1))
    return peak_st * (0.5 - 0.5 * np.cos(np.pi * w))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def lift(x: np.ndarray, rate: int, peak_st: float) -> tuple[np.ndarray, dict]:
    """Return (audio, report). The audio has the same length and loudness as the input."""
    peak = float(np.clip(peak_st, 0.0, MAX_LIFT_ST))
    if peak == 0.0:
        return x, {"applied": False, "reason": "no lift requested"}
    if len(x) < rate // 2:
        return x, {"applied": False, "reason": "clip too short"}
    try:
        x64 = x.astype(np.float64)
        f0, t = pw.harvest(x64, rate, f0_floor=70.0, f0_ceil=400.0, frame_period=FRAME_MS)
        voiced = f0 > 0
        if voiced.mean() < MIN_VOICED:
            return x, {"applied": False, "reason": f"only {voiced.mean():.0%} voiced"}
        sp = pw.cheaptrick(x64, f0, t, rate)
        ap = pw.d4c(x64, f0, t, rate)
        st = contour(t, peak)
        y = pw.synthesize(f0 * 2 ** (st / 12), sp, ap, rate, FRAME_MS)
    except Exception as e:  # noqa: BLE001 - a cosmetic effect must never fail the render
        return x, {"applied": False, "reason": f"{type(e).__name__}: {e}"}
    y = y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    y = y * min(2.0, max(0.5, _rms(x) / (_rms(y) or 1.0)))  # keep the original loudness
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    return y, {
        "applied": True, "peak_st": round(peak, 2), "voiced_fraction": round(float(voiced.mean()), 2),
        "f0_before_hz": round(float(np.median(f0[voiced])), 1),
        "f0_target_hz": round(float(np.median((f0 * 2 ** (st / 12))[voiced])), 1),
    }
