#!/usr/bin/env python3
"""cw_quality.py - the CW copy-quality measuring apparatus.

The Morse analog of the STVT MER-dial and the FM quality meter: it measures how
well we are COPYING a CW signal, and diagnoses WHY when copy fails - all from the
signal, without needing to know what was sent (so a confident-but-wrong decode
can't fake it, the way q_ratio does).

The headline number is the EYE-OPENING Q - the decision margin between the
key-down and key-up envelope levels:

    Q = (mu_on - mu_off) / (sigma_on + sigma_off)     # CW's eye-diagram opening
    Pe ~= 0.5 * erfc(Q / sqrt(2))                      # -> bit-error probability

Q is measured BEFORE any decode decision, so it's honest. It's the CW twin of
constellation MER / eye height in digital comms. Backed by three lines of
evidence that all agree: RSCW/CW-Skimmer/fldigi practice (matched-filter +
soft decision), the eye-opening = decision-margin theory, and our own corpus
(clean KI4XH -> Q 6.3; fading garbage 'BKTUTOM' -> Q 2.1).

Layers measured, high to low:
  RF      : tone SNR in the CW bandwidth (dB)
  SIGNAL  : eye-opening Q (dB) + implied copy % - the honest headline
  TIMING  : dit/dah cluster separation + WPM + sender 'fist' (straight/bug/keyer)
  CHANNEL : fade depth/rate, noise kurtosis (QRN), in-band peak count (QRM),
            frequency drift/chirp - the failure-class diagnosis

Usage:
  python cw_quality.py <capture.cs16> [--off HZ] [--fs 250000]
  python cw_quality.py --corpus                 # score the whole corpus
"""
import argparse
import glob
import sys
from math import erfc, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cw

FS = 250_000.0
CORPUS = HERE.parent / "lab" / "cw_corpus"

# copy-quality cliffs (Q eye-opening), calibrated on the corpus + Pe(Q):
#   Q >= 4.0  solid copy      (Pe < 1e-4)
#   Q ~  3.0  readable        (Pe ~ 1e-2, the watchability knee)
#   Q <  2.5  failing/garbage (Pe > 5e-2, hard-threshold decoders break here)
Q_SOLID, Q_READABLE = 4.0, 3.0


def _rle(on):
    runs = []
    cur = bool(on[0]); ln = 1
    for v in on[1:]:
        if bool(v) == cur:
            ln += 1
        else:
            runs.append((cur, ln)); cur, ln = bool(v), 1
    runs.append((cur, ln))
    return runs


def tone_snr_db(iq, off, fs, cw_bw=150.0):
    """Signal-minus-noise power in the CW passband vs the noise floor in the
    shoulders. cw_bw ~ occupied bandwidth of hand CW (research: ~K*baud, 100-150 Hz)."""
    N = 1 << 14
    m = len(iq) // N * N
    if m < N:
        return 0.0
    seg = iq[:m].reshape(-1, N) * np.hanning(N).astype(np.float32)
    P = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2).mean(0)
    binhz = fs / N
    c = N // 2 + int(round(off / binhz))
    half = max(1, int(cw_bw / binhz))
    c = min(max(c, half + 1), len(P) - half - 1)
    sig = P[c - half:c + half + 1].sum()
    lo, hi = int(2000 / binhz), int(6000 / binhz)
    noiseband = np.concatenate([P[max(0, c - hi):c - lo], P[c + lo:min(len(P), c + hi)]])
    npb = np.median(noiseband) if len(noiseband) else 1e-12
    noise_in_bw = npb * (2 * half + 1)
    return float(10 * np.log10(max(sig - noise_in_bw, 1e-12) / max(noise_in_bw, 1e-12)))


def eye_opening(env):
    """The headline metric. Otsu-split the envelope into key-off/key-on clusters
    (pre-decision), then the normalized decision margin Q and its implied copy %.
    Returns (Q, Q_dB, copy_pct, thr, mu_off, mu_on)."""
    e = env.astype(np.float64)
    e = e[e > 0]
    if len(e) < 200:
        return 0.0, -30.0, 0.0, 0.0, 0.0, 0.0
    hist, edges = np.histogram(e, bins=80)
    p = hist / hist.sum()
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p); w1 = 1 - w0
    m0 = np.cumsum(p * mids) / np.maximum(w0, 1e-12)
    mT = (p * mids).sum()
    m1 = (mT - np.cumsum(p * mids)) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    thr = mids[int(np.argmax(between))]
    on, off = e[e > thr], e[e <= thr]
    if len(on) < 20 or len(off) < 20:
        return 0.0, -30.0, 0.0, thr, 0.0, 0.0
    mon, moff = on.mean(), off.mean()
    son, soff = on.std(), off.std()
    Q = (mon - moff) / (son + soff + 1e-12)
    # copy% as a sigmoid on Q anchored at the corpus-calibrated cliffs (per-sample
    # Pe(Q) is far too optimistic - envelope samples are correlated and the real
    # copy-killer is TIMING error, not per-sample slicing - so use the empirical
    # eye->readability curve: Q~2.1 garbage ~35%, Q3 readable ~70%, Q4 solid ~90%)
    copy_pct = 100.0 / (1.0 + np.exp(-(Q - 2.5) * 1.5))
    q_db = 20 * np.log10(max(Q, 1e-3))
    return float(Q), float(q_db), float(copy_pct), float(thr), float(moff), float(mon)


def timing_and_fist(env, aud, thr):
    """dit/dah cluster separation, WPM, and the sender's 'fist' (straight key /
    bug / keyer / machine) from per-element-type jitter (research agent 1)."""
    on = env > thr
    runs = _rle(on)
    onr = np.array([l for s, l in runs if s], float)
    # reject sub-dit fragments: QSB/noise chops key-down into tiny blips that
    # poison the clustering (the 1828-wpm bug). Keep runs >= 18 ms (60 wpm dit)
    # and <= 500 ms; a real dit at 5-40 wpm is 30-240 ms.
    floor, ceil = 0.018 * aud, 0.5 * aud
    onr = onr[(onr >= floor) & (onr <= ceil)]
    if len(onr) < 8:
        return dict(wpm=0, dahdit=0, sep=0, fist="(too few clean elements)")
    lu = np.log(onr)
    c = np.array([lu.min(), lu.max()])           # log-space k-means, k=2
    for _ in range(25):
        lab = (np.abs(lu[:, None] - c[None, :])).argmin(1)
        nc = np.array([lu[lab == j].mean() if np.any(lab == j) else c[j] for j in (0, 1)])
        if np.allclose(nc, c):
            break
        c = nc
    dits, dahs = onr[lab == np.argmin(c)], onr[lab == np.argmax(c)]
    if len(dits) < 2 or len(dahs) < 2:
        return dict(wpm=round(1.2 / (np.median(onr) / aud), 1), dahdit=0, sep=0, fist="(unresolved)")
    dit_u, dah_u = dits.mean(), dahs.mean()
    dit_cv, dah_cv = dits.std() / dit_u, dahs.std() / dah_u
    ratio = dah_u / dit_u
    sep = (np.log(dah_u) - np.log(dit_u)) / (np.std(np.log(dits)) + np.std(np.log(dahs)) + 1e-9)
    wpm = 1.2 / (dit_u / aud)
    # fist classification (agent-1 fingerprint table)
    if dit_cv < 0.10 and dah_cv < 0.12 and abs(ratio - 3.0) < 0.5:
        fist = "keyer/machine (precise 1:3)"
    elif dit_cv < 0.12 and ratio > 3.3:
        fist = "bug (tight dits, long dahs)"
    elif dit_cv > 0.18 or dah_cv > 0.28:
        fist = "straight key (hand jitter)"
    else:
        fist = "electronic keyer"
    return dict(wpm=round(float(wpm), 1), dahdit=round(float(ratio), 2),
                sep=round(float(sep), 2), dit_cv=round(float(dit_cv), 2),
                dah_cv=round(float(dah_cv), 2), fist=fist)


def channel_health(iq, env, aud, off, fs, thr, wpm):
    """Failure-class fingerprints: fade depth/rate (QSB), noise kurtosis (QRN),
    in-band peak count (QRM), tone-frequency spread (drift/chirp/Doppler)."""
    out = {}
    on = env > thr
    runs = _rle(on)
    # --- fade: per-ON-run mark level over time (samples the fade at element rate)
    idx = 0; marks = []
    for s, ln in runs:
        if s and ln > 2:
            marks.append(float(np.median(env[idx:idx + ln])))
        idx += ln
    marks = np.array(marks)
    if len(marks) >= 6:
        p10, p90 = np.percentile(marks, 10), np.percentile(marks, 90)
        out["fade_depth_db"] = round(float(20 * np.log10((p90 + 1e-9) / (p10 + 1e-9))), 1)
        # fast vs slow: correlation between adjacent element mark levels
        a = marks[:-1] - marks[:-1].mean(); b = marks[1:] - marks[1:].mean()
        r = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
        out["fade_fast"] = r < 0.3 and out["fade_depth_db"] > 6   # flutter: uncorrelated + deep
    else:
        out["fade_depth_db"] = 0.0; out["fade_fast"] = False
    # --- QRN: excess kurtosis of the wideband magnitude during key-up (noise only)
    off_mag = env[~on]
    if len(off_mag) > 200:
        x = off_mag - off_mag.mean()
        k = float((x ** 4).mean() / ((x ** 2).mean() ** 2 + 1e-12) - 3.0)
        out["kurtosis"] = round(k, 1)
    else:
        out["kurtosis"] = 0.0
    # --- QRM: distinct persistent tones within +/-1.5 kHz of our tone
    N = 1 << 13
    m = len(iq) // N * N
    if m >= N:
        seg = iq[:m].reshape(-1, N) * np.hanning(N).astype(np.float32)
        P = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2).mean(0)
        binhz = fs / N; c = N // 2 + int(round(off / binhz))
        span = int(1500 / binhz)
        band = P[max(0, c - span):c + span]
        if len(band):
            db = 10 * np.log10(band / np.median(band) + 1e-12)
            peaks = [i for i in range(2, len(db) - 2)
                     if db[i] > 10 and db[i] >= db[i - 1] and db[i] > db[i + 1]]
            out["inband_tones"] = len(peaks)
    out.setdefault("inband_tones", 1)
    # --- coherence vs element length (flutter danger flag)
    out["elem_ms"] = round(1200.0 / wpm, 1) if wpm else 0.0
    return out


def diagnose(m):
    """Turn the measurement vector into a plain-English verdict + the failure
    class, the way the MER-dial names TV failure modes by their signature."""
    Q, snr = m["eye_q"], m["snr_db"]
    ch = m["channel"]
    if Q >= Q_SOLID:
        verdict = "SOLID COPY"
    elif Q >= Q_READABLE:
        verdict = "READABLE"
    else:
        verdict = "FAILING"
    # failure class (agent-2 decision tree): QRN -> QRM -> QSB/flutter -> freq.
    # Only meaningful when copy is degraded; a SOLID copy has no failure to name.
    cls = []
    if verdict != "SOLID COPY":
        if ch.get("kurtosis", 0) > 4 and ch.get("inband_tones", 1) <= 1:
            cls.append("QRN (impulsive noise / static crashes -> false dits)")
        if ch.get("inband_tones", 1) >= 2:
            cls.append(f"QRM ({ch['inband_tones']} signals in passband -> interference)")
        if ch.get("fade_fast"):
            cls.append(f"FLUTTER / fast QSB ({ch.get('fade_depth_db',0)} dB, faster than a character)")
        elif ch.get("fade_depth_db", 0) > 8:
            cls.append(f"QSB (slow fading {ch.get('fade_depth_db',0)} dB)")
    # the honest headline reasoning
    if verdict == "FAILING":
        if snr < 6 and not cls:
            why = "signal too weak (low SNR) - the eye is closed by noise"
        elif snr >= 15 and Q < Q_READABLE and not cls:
            why = "LOUD but the eye is closed - keying is smeared (fading/multipath), not a level problem"
        elif cls:
            why = "; ".join(cls)
        else:
            why = "eye closed - marginal copy"
    elif verdict == "READABLE":
        why = ("clean but " + "; ".join(cls)) if cls else "weak but clean - copyable"
    else:
        why = "clean signal, open eye" + (" (" + "; ".join(cls) + ")" if cls else "")
    return verdict, why, cls


def measure(iq, fs=FS, off=None):
    """Full apparatus: measure every layer + a composite copy score + diagnosis."""
    if off is None:
        off = cw.find_offset(iq, fs, 50000)
    snr = tone_snr_db(iq, off, fs)
    env, aud = cw.envelope(iq, fs, off)
    Q, q_db, copy_pct, thr, moff, mon = eye_opening(env)
    tim = timing_and_fist(env, aud, thr)
    ch = channel_health(iq, env, aud, off, fs, thr, tim.get("wpm", 0) or 20)
    m = dict(off_hz=round(float(off), 1), snr_db=round(snr, 1),
             eye_q=round(Q, 2), eye_db=round(q_db, 1), copy_pct=round(copy_pct, 0),
             wpm=tim.get("wpm", 0), dahdit=tim.get("dahdit", 0),
             timing_sep=tim.get("sep", 0), fist=tim.get("fist", ""),
             channel=ch)
    m["verdict"], m["why"], m["failure_class"] = diagnose(m)
    # composite copy score 0-100: eye-opening dominates (it predicts Pe), SNR gates
    eye_score = 100 * min(1.0, max(0.0, (Q - 1.5) / (Q_SOLID - 1.5)))
    snr_gate = min(1.0, max(0.0, (snr - 3) / 15.0))
    m["copy_score"] = round(float(eye_score * (0.4 + 0.6 * snr_gate)), 0)
    return m


def _load(f):
    raw = np.fromfile(f, np.int16)
    return ((raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)) / 32768.0).astype(np.complex64)


def _print(name, m):
    print(f"\n=== {name} ===")
    print(f"  VERDICT : {m['verdict']}  (copy score {m['copy_score']:.0f}/100)")
    print(f"  why     : {m['why']}")
    print(f"  RF      : tone SNR {m['snr_db']} dB")
    print(f"  SIGNAL  : eye-opening Q {m['eye_q']} ({m['eye_db']} dB)  implied copy {m['copy_pct']:.0f}%")
    print(f"  TIMING  : {m['wpm']} wpm  dah:dit {m['dahdit']}  sep {m['timing_sep']}  -> {m['fist']}")
    ch = m["channel"]
    print(f"  CHANNEL : fade {ch.get('fade_depth_db',0)} dB{' (FLUTTER)' if ch.get('fade_fast') else ''}"
          f"  kurtosis {ch.get('kurtosis',0)}  in-band tones {ch.get('inband_tones',1)}"
          f"  element {ch.get('elem_ms',0)} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?")
    ap.add_argument("--off", type=float, default=None)
    ap.add_argument("--fs", type=float, default=FS)
    ap.add_argument("--corpus", action="store_true")
    a = ap.parse_args()
    if a.corpus:
        for f in sorted(glob.glob(str(CORPUS / "*.cs16"))):
            _print(Path(f).name, measure(_load(f), a.fs))
    elif a.file:
        _print(Path(a.file).name, measure(_load(a.file), a.fs, a.off))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
