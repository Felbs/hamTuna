#!/usr/bin/env python3
"""cw_synth.py - synthetic labeled CW generator (the training/test foundation).

Supervised decoding needs (signal, KNOWN-text) pairs; you can't get labels off the
air. This makes them: ham-realistic text -> Morse envelope at chosen WPM / fist /
noise / QSB, with the text as the label. Uses:
  - a labeled regression/test set for the decoder + LM (unlimited, known answers)
  - realistic training data for a future neural decoder (inject REAL harvested
    band-noise via add_noise_profile() to close the sim-to-real gap)

Deterministic given a seed (Date/random-free constraints: pass seeds explicitly).

  python cw_synth.py demo          # print a few generated labels
  python cw_synth.py testset 30    # build 30 labeled envelopes, self-decode acc
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cw

PREFIXES = ["W", "K", "N", "AA", "KA", "KB", "KC", "KI", "KK", "W1", "W4", "VE", "G", "DL"]
SUFFIXES = ["AW", "XH", "DHW", "ABC", "QRP", "TT", "JD", "OM", "RN", "SK", "PZ"]
WORDS = ["THE", "AND", "THAT", "WAS", "NOTHING", "OF", "KIND", "TOLD", "THEM", "MY",
         "STORY", "GUD", "FB", "TNX", "RIG", "ANT", "WX", "HR", "COLD", "SNOWY",
         "NAME", "IS", "JOHN", "BOB", "JIM", "SOLID", "COPY", "SIGNAL", "GL", "CUL"]
QCODES = ["QRZ", "QSB", "QRM", "QTH", "QSL", "QRP", "QSO"]


def _call(rng):
    return rng.choice(PREFIXES) + str(rng.integers(0, 10)) + rng.choice(SUFFIXES)


def random_text(rng, kind=None):
    """A ham-realistic transmission (label)."""
    kind = kind or rng.choice(["cq", "exchange", "ragchew", "call"])
    if kind == "cq":
        return f"CQ CQ CQ DE {_call(rng)} {_call(rng)} K"
    if kind == "call":
        return f"{_call(rng)} DE {_call(rng)} {_call(rng)} K"
    if kind == "exchange":
        return f"{_call(rng)} DE {_call(rng)} UR RST 5NN 5NN {rng.choice(QCODES)} K"
    n = int(rng.integers(4, 9))
    return " ".join(rng.choice(WORDS) for _ in range(n))


def render(text, wpm=20, fs=8000.0, jitter=0.10, weight=1.0, noise=0.12,
           fade=0.0, fade_hz=0.3, rise_ms=5.0, qrn=0.0, seed=0):
    """text -> magnitude envelope (aud=fs). Domain-randomized for ANTENNA
    UNIVERSALITY: jitter=timing spread, weight=fist (dah:dit), fade=QSB depth,
    rise_ms=raised-cosine keying edges (~5ms real; anti-click), qrn=impulse-noise
    rate (static crashes). The envelope is what the net sees, and it's gain-
    normalized downstream, so spanning these = works on any antenna. Returns (env,fs)."""
    rng = np.random.default_rng(seed)
    u = 1.2 / wpm * fs
    parts = []
    for ch in text.upper():
        if ch == " ":
            parts.append((0, 7 * u * (1 + rng.normal(0, jitter)))); continue
        code = cw.INV.get(ch)
        if not code:
            continue
        for ei, el in enumerate(code):
            L = (1 if el == "." else 3 * weight) * u * (1 + rng.normal(0, jitter))
            parts.append((1, max(3, L)))
            if ei < len(code) - 1:
                parts.append((0, u * (1 + rng.normal(0, jitter))))
        parts.append((0, 3 * u * (1 + rng.normal(0, jitter))))       # letter gap
    key = np.concatenate([np.full(int(max(1, d)), float(lvl)) for lvl, d in parts]).astype(np.float32)
    # raised-cosine rise/fall so edges aren't perfectly square (real keying shaping)
    r = max(1, int(rise_ms * 1e-3 * fs))
    if r > 1:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(r) / r)).astype(np.float32)
        k = np.ones(r, np.float32) / r
        key = np.convolve(key, k, mode="same")           # smooth edges ~ raised cosine
    sig = key
    if fade > 0:                                          # QSB
        t = np.arange(len(sig)) / fs
        sig = sig * (1 - fade * 0.5 * (1 + np.sin(2 * np.pi * fade_hz * t))).astype(np.float32)
    # RICIAN/RAYLEIGH statistics: real envelopes are |signal + COMPLEX noise|, not
    # signal + real-gaussian. This matches what cw.envelope produces on real IQ.
    n_re = rng.normal(0, noise, len(sig)).astype(np.float32)
    n_im = rng.normal(0, noise, len(sig)).astype(np.float32)
    env = np.sqrt((sig + n_re) ** 2 + n_im ** 2)
    if qrn > 0:                                           # impulsive static crashes
        n_imp = rng.poisson(qrn * len(env) / fs)
        for _ in range(int(n_imp)):
            p = int(rng.integers(0, len(env)))
            w = int(rng.integers(1, max(2, int(0.01 * fs))))
            env[p:p + w] += rng.uniform(1, 4)
    # match cw.envelope's 8 ms boxcar smoothing (magnitude THEN smooth, as on real IQ)
    ks = max(1, int(0.008 * fs))
    env = np.convolve(env, np.ones(ks, np.float32) / ks, mode="same")
    return env.astype(np.float32), fs


def render_iq(text, wpm=20, fs_iq=48000.0, f0=700.0, jitter=0.10, weight=1.0,
              noise=0.12, fade=0.0, fade_hz=0.3, rise_ms=5.0, seed=0):
    """text -> complex IQ (tone f0 keyed on/off + COMPLEX noise) at fs_iq. Run this
    through cw.envelope() exactly like a real capture -> identical train/test
    processing (the EXP-4 exact-match path). Returns (iq, fs_iq, f0)."""
    rng = np.random.default_rng(seed)
    u = 1.2 / wpm * fs_iq
    parts = []
    for ch in text.upper():
        if ch == " ":
            parts.append((0, 7 * u * (1 + rng.normal(0, jitter)))); continue
        code = cw.INV.get(ch)
        if not code:
            continue
        for ei, el in enumerate(code):
            L = (1 if el == "." else 3 * weight) * u * (1 + rng.normal(0, jitter))
            parts.append((1, max(3, L)))
            if ei < len(code) - 1:
                parts.append((0, u * (1 + rng.normal(0, jitter))))
        parts.append((0, 3 * u * (1 + rng.normal(0, jitter))))
    key = np.concatenate([np.full(int(max(1, d)), float(lvl)) for lvl, d in parts]).astype(np.float32)
    r = max(1, int(rise_ms * 1e-3 * fs_iq))
    if r > 1:
        key = np.convolve(key, np.ones(r, np.float32) / r, mode="same")
    if fade > 0:
        t = np.arange(len(key)) / fs_iq
        key = key * (1 - fade * 0.5 * (1 + np.sin(2 * np.pi * fade_hz * t))).astype(np.float32)
    t = np.arange(len(key))
    tone = key * np.exp(2j * np.pi * f0 / fs_iq * t)
    n = (rng.normal(0, noise, len(key)) + 1j * rng.normal(0, noise, len(key)))
    return (tone + n).astype(np.complex64), fs_iq, f0


def add_noise_profile(env, noise_samples, gain=1.0):
    """Inject REAL harvested band-noise (from a capture's key-up intervals) into a
    synthetic envelope - closes the sim-to-real gap AG1LE's generator missed."""
    if len(noise_samples) == 0:
        return env
    reps = int(np.ceil(len(env) / len(noise_samples)))
    n = np.tile(noise_samples, reps)[:len(env)]
    return np.abs(env + gain * n.astype(np.float32))


def dataset(n=20, seed=0, **render_kw):
    """n labeled (env, text) pairs spanning WPM/noise/fade (deterministic)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        text = random_text(rng)
        kw = dict(wpm=int(rng.integers(12, 32)), jitter=float(rng.uniform(0.05, 0.2)),
                  noise=float(rng.uniform(0.08, 0.3)),
                  fade=float(rng.choice([0.0, 0.0, 0.4, 0.6])), seed=int(rng.integers(1, 1e6)))
        kw.update(render_kw)
        env, fs = render(text, **kw)
        out.append((env, fs, text, kw))
    return out


def _cer(a, b):
    """char error rate (Levenshtein / len)."""
    a, b = a.replace(" ", ""), b.replace(" ", "")
    if not b:
        return 1.0
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)] / len(b)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        rng = np.random.default_rng(7)
        for _ in range(6):
            print("  " + random_text(rng))
    elif cmd == "testset":
        import cw_bayes, cw_lm
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        ds = dataset(n, seed=3)
        cers = []
        for env, fs, text, kw in ds:
            dec = cw_lm.rescore(cw_bayes.decode_bayes(env, fs, soft=(kw["fade"] > 0))[0])
            cers.append(_cer(dec, text))
        cers = np.array(cers)
        print(f"synthetic test set n={n}: mean CER={cers.mean():.2f}  "
              f"clean(CER<0.2)={int((cers < 0.2).sum())}/{n}")
