"""cw.py - hamTuna: Morse / CW decoder (ham CW + NDB aviation beacons).

The oldest digital mode. On-off keying of a carrier: dit, dah, gaps.
Decoding it turns an IDENTIFIED beacon (we see the carrier) into a
DECODED one (we read its callsign). NDB beacons (190-535 kHz) key their
2-3 letter ID continuously - legal, public, and a clean first target.

Pipeline: mix the carrier to DC -> envelope -> adaptive on/off threshold
-> run-length -> dit/dah/gap classification (self-calibrating WPM) ->
Morse -> text.

Modes:
  selftest - synthesize "NDB" in Morse, add noise, decode it back
  decode   - decode a capture file (cs16) at a given carrier offset

Example:  python cw.py decode --file cap.cs16 --offset -6200
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

MORSE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
    "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z", "-----": "0", ".----": "1", "..---": "2",
    "...--": "3", "....-": "4", ".....": "5", "-....": "6", "--...": "7",
    "---..": "8", "----.": "9", "-.-.--": "!", "-..-.": "/"}
INV = {v: k for k, v in MORSE.items()}


def envelope(iq, fs, off_hz, aud=8000):
    from scipy.signal import resample_poly
    from math import gcd
    n = np.arange(len(iq), dtype=np.float64)
    x = iq * np.exp(-2j * np.pi * off_hz / fs * n)
    g = gcd(int(aud), int(fs))
    x = resample_poly(x, int(aud) // g, int(fs) // g).astype(np.complex64)
    env = np.abs(x).astype(np.float32)
    k = max(1, int(aud * 0.008))          # 8 ms smoother
    return np.convolve(env, np.ones(k, np.float32) / k, mode="same"), aud


def envelope_coherent(iq, fs, off_hz, aud=8000, coh_ms=12.0,
                      search_hz=0.0, bin_hz=8.0):
    """Coherent front-end (EXP-13): integrate the COMPLEX baseband over a short
    matched window BEFORE taking magnitude - |sum(x)| instead of the classic
    sum(|x|). For a phase-stable tone this gains ~sqrt(window) of SNR.

    *** KILLED, kept only for study (exp13_coherent.py). It is a SYNTHETIC-BENCH
    MIRAGE: on synthetic IQ it crushes incoherent (noise-floor 1.23 -> 2.50), but
    on REAL captures it collapses (callsign recall 0.257 -> 0.03-0.06 at EVERY
    window 2-12 ms). Real HF CW has phase noise, drift and QSB that decorrelate
    coherent integration; the incoherent envelope's phase-blindness is a FEATURE.
    DO NOT wire this into decode_env_* / the panel - it destroys real copy. ***"""
    from scipy.signal import resample_poly
    from math import gcd
    n = np.arange(len(iq), dtype=np.float64)
    x = iq * np.exp(-2j * np.pi * off_hz / fs * n)
    g = gcd(int(aud), int(fs))
    x = resample_poly(x, int(aud) // g, int(fs) // g).astype(np.complex64)
    k = max(1, int(aud * coh_ms / 1000.0))
    box = np.ones(k, np.float32) / k
    m = np.arange(len(x), dtype=np.float64)
    nb = int(search_hz // bin_hz) if search_hz > 0 else 0
    best = None
    for b in range(-nb, nb + 1):
        xb = x if b == 0 else x * np.exp(-2j * np.pi * (b * bin_hz) / aud * m)
        mag = np.abs(np.convolve(xb, box, mode="same")).astype(np.float32)
        best = mag if best is None else np.maximum(best, mag)
    return best, aud


def envelope2(iq, fs, off_hz, aud=8000, bw_hz=150):
    """EXP H10: like envelope() but with a NARROW complex low-pass around DC BEFORE
    the magnitude detector. envelope() detects over the full +/-aud/2 (~4 kHz) noise
    bandwidth; a CW signal is only ~100-150 Hz wide, so band-limiting to +/-bw_hz
    cuts noise power by ~10*log10((aud/2)/bw_hz) (~14 dB at 150 Hz) before |x| - the
    classic weak-CW narrow-filter SNR win. Too-narrow smears fast dits, so bw is
    swept. Same (env, aud) contract as envelope()."""
    from scipy.signal import resample_poly, firwin
    from math import gcd
    n = np.arange(len(iq), dtype=np.float64)
    x = iq * np.exp(-2j * np.pi * off_hz / fs * n)
    g = gcd(int(aud), int(fs))
    x = resample_poly(x, int(aud) // g, int(fs) // g).astype(np.complex64)
    if bw_hz and bw_hz < aud / 2:
        ntaps = int(np.clip(aud / bw_hz, 21, 301)) | 1        # odd; ~1 cycle of bw
        taps = firwin(ntaps, bw_hz / (aud / 2)).astype(np.float32)
        x = np.convolve(x, taps, mode="same").astype(np.complex64)   # complex LP around DC
    env = np.abs(x).astype(np.float32)
    k = max(1, int(aud * 0.008))          # 8 ms smoother
    return np.convolve(env, np.ones(k, np.float32) / k, mode="same"), aud


def _gap_boundaries(off_runs, dit):
    """Adaptive intra/letter and letter/word gap boundaries via 3-means on the
    internal gaps. Fixed 2*dit / 5*dit thresholds break under FARNSWORTH spacing
    (fast characters, stretched gaps): the letter-gap cluster stretches past
    5*dit and gets misread as word gaps, splitting callsigns ('KI 4XH'). Fitting
    the actual gap clusters puts the word boundary in the real valley between
    them. Falls back to the classic multiples when the gaps are too few/degenerate.
    """
    g = np.array([x for x in off_runs if x < 25 * dit], float)   # drop inter-tx silence
    if len(g) < 6:
        return 2 * dit, 5 * dit
    c = np.array([1.0, 3.5, 7.0]) * dit                          # init at 1:3:7 units
    for _ in range(30):
        lab = np.abs(g[:, None] - c[None, :]).argmin(1)
        newc = np.array([g[lab == k].mean() if np.any(lab == k) else c[k]
                         for k in range(3)])
        newc.sort()
        if np.allclose(newc, c):
            break
        c = newc
    if c[2] < 1.6 * c[1]:                        # letter & word clusters not separated
        lb = np.sqrt(c[0] * c[1]) if c[1] > 1.3 * c[0] else 2 * dit
        return lb, 1e18                          # -> no word gaps in this stream
    lb, wb = np.sqrt(c[0] * c[1]), np.sqrt(c[1] * c[2])  # geometric-mean valleys
    if not (dit < lb < wb):
        return 2 * dit, 5 * dit
    return lb, wb


def _runs_to_text(runs, aud):
    """Shared back end: run-length list -> self-calibrated Morse text + info.
    Used by both the classic threshold decoder and the matched-filter decoder."""
    on_runs = [ln for s, ln in runs if s]
    if len(on_runs) < 3:
        return "", {"runs": len(runs)}
    # dit length = the shorter cluster of ON runs (k-means-lite, 2 groups)
    o = np.array(on_runs, float)
    med = np.median(o)
    dit = np.median(o[o <= med]) or med
    # gap boundaries adapt to the stream's own spacing (Farnsworth-safe)
    off_runs = [ln for i, (s, ln) in enumerate(runs) if not s and 0 < i < len(runs) - 1]
    lb, wb = _gap_boundaries(off_runs, dit)
    # collect per-letter Morse + a per-element confidence (distance of each ON run
    # from the dit/dah decision line, normalized) so the panel can show the
    # dits-and-dashes alongside the English (educational) and flag shaky elements.
    text, tokens = [], []
    sym, conf = "", []

    def _emit():
        if not sym:
            return
        text.append(MORSE.get(sym, "?"))
        # letter quality = 1 - the WORST element's ambiguity (1.0 clean/on a cluster
        # center, ->0 when an element straddles the dit/dah line). Panel can dim
        # low-q letters so you SEE where the copy got shaky.
        tokens.append({"m": sym, "c": MORSE.get(sym, "?"),
                       "q": round(max(0.0, 1.0 - max(conf)), 2) if conf else 1.0})
        conf.clear()

    for s, ln in runs:
        if s:                              # tone
            sym += "-" if ln > 2 * dit else "."
            # element ambiguity = normalized distance to the nearer cluster center
            # (0 = clean dit/dah, ~1 = right on the 2*dit boundary)
            conf.append(float(min(abs(ln - dit), abs(ln - 3 * dit)) / max(dit, 1e-9)))
        else:                              # gap
            if ln > wb:                    # word gap
                _emit()
                text.append(" ")
                tokens.append({"m": "/", "c": " ", "q": 1.0})
                sym = ""
            elif ln > lb:                  # letter gap
                _emit()
                sym = ""
    _emit()
    return "".join(text).strip(), {"dit_ms": round(1000 * dit / aud, 1),
                                   "wpm": round(1.2 / (dit / aud), 1),
                                   "elements": len(on_runs),
                                   "morse": " ".join(t["m"] for t in tokens),
                                   "tokens": tokens}


def _runs_to_text2(runs, aud):
    """EXP H5 (graceful degradation): like _runs_to_text but emits '?' for a LETTER
    only when a dit/dah call is genuinely ambiguous - an ON run straddling the
    2*dit boundary AND resolving it both ways yields DIFFERENT valid letters.
    Clean copy has no straddling elements -> zero spurious '?'; weak copy degrades
    to '?' (honest 'I missed that') instead of a confident WRONG letter."""
    on_runs = [ln for s, ln in runs if s]
    if len(on_runs) < 3:
        return "", {"runs": len(runs)}
    o = np.array(on_runs, float)
    med = np.median(o)
    dit = np.median(o[o <= med]) or med
    off_runs = [ln for i, (s, ln) in enumerate(runs) if not s and 0 < i < len(runs) - 1]
    lb, wb = _gap_boundaries(off_runs, dit)
    lo_a, hi_a = 1.5 * dit, 2.5 * dit          # ambiguity band around the 2*dit boundary

    def _flush(elems):
        if not elems:
            return ""
        amb = [i for i, (_, a) in enumerate(elems) if a]
        cands = set()
        if len(amb) <= 3:                       # enumerate flips of ambiguous elements
            for mask in range(1 << len(amb)):
                chars = [c for c, _ in elems]
                for b, i in enumerate(amb):
                    if mask & (1 << b):
                        chars[i] = "-" if chars[i] == "." else "."
                lt = MORSE.get("".join(chars))
                if lt:
                    cands.add(lt)
        else:
            lt = MORSE.get("".join(c for c, _ in elems))
            cands = {lt} if lt else set()
        return next(iter(cands)) if len(cands) == 1 else "?"

    text, elems = [], []
    for s, ln in runs:
        if s:
            elems.append(("-" if ln > 2 * dit else ".", lo_a < ln < hi_a))
        elif ln > wb:                           # word gap
            if elems:
                text.append(_flush(elems))
            text.append(" "); elems = []
        elif ln > lb:                           # letter gap
            if elems:
                text.append(_flush(elems)); elems = []
    if elems:
        text.append(_flush(elems))
    return "".join(text).strip(), {"dit_ms": round(1000 * dit / aud, 1),
                                   "wpm": round(1.2 / (dit / aud), 1),
                                   "elements": len(on_runs), "graceful": True}


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


def decode_env(env, aud):
    """Adaptive on/off -> run lengths -> self-calibrated Morse (classic path)."""
    hi, lo = np.percentile(env, 90), np.percentile(env, 25)
    if hi - lo < 1e-6:
        return "", {}
    thr = lo + 0.4 * (hi - lo)
    return _runs_to_text(_rle(env > thr), aud)


def decode_env_mf(env, aud):
    """Matched-filter + fade-tracking decoder (the 'superior math' path).

    Two research-backed wins over the classic global threshold:
      1. MATCHED FILTER - a boxcar of one dit is the SNR-optimal detector for a
         rectangular OOK element in AWGN (RSCW/fldigi). It integrates each element
         and suppresses noise before slicing.
      2. FADE-TRACKING THRESHOLD - instead of ONE threshold for the whole capture
         (which a QSB fade sinks the signal below, mid-character), the slice level
         rides the signal up and down in ~0.6 s blocks, with hysteresis to stop
         edge chatter. This is what copies through the 'loud but eye closed'
         fading that breaks the classic decoder.
    Falls back to the classic result if it can't get a coarse dit estimate."""
    if len(env) < aud // 2:
        return decode_env(env, aud)
    # coarse dit from a quick global threshold, to size the matched filter
    hi, lo = np.percentile(env, 90), np.percentile(env, 25)
    if hi - lo < 1e-6:
        return "", {}
    on0 = env > (lo + 0.4 * (hi - lo))
    on0_runs = [ln for s, ln in _rle(on0) if s]
    if len(on0_runs) < 3:
        return "", {"runs": len(on0_runs)}
    o = np.array(on0_runs, float)
    dit = np.median(o[o <= np.median(o)]) or np.median(o)
    ndit = int(np.clip(dit, aud * 0.015, aud * 0.3))           # matched-filter width
    # 1) matched filter: integrate-and-dump over one dit (boxcar)
    mf = np.convolve(env, np.ones(ndit, np.float32) / ndit, mode="same")
    # 2) fade-tracking threshold: per-block on/off levels, interpolated per-sample
    blk = max(int(0.6 * aud), 4 * ndit)
    gfloor = np.percentile(mf, 60)                              # global 'is there signal' floor
    centers, levels, margins = [], [], []
    for b in range(0, len(mf), blk):
        seg = mf[b:b + blk]
        if len(seg) < ndit:
            continue
        blo, bhi = np.percentile(seg, 25), np.percentile(seg, 92)
        centers.append(b + len(seg) / 2)
        # per-block SNR gate: if the local eye is closed (bhi barely above blo)
        # or the block is all noise (bhi below the global signal floor), force the
        # threshold sky-high so noise-only regions emit NO elements (kills the
        # 'IN T'/'?EHI?E' artifacts the naive tracker slices out of noise).
        if bhi < gfloor or bhi < 1.6 * blo:
            levels.append(bhi * 5 + 1e-6); margins.append(0.0)
        else:
            levels.append(blo + 0.5 * (bhi - blo)); margins.append(0.5 * (bhi - blo))
    if len(centers) < 2:
        return _runs_to_text(_rle(mf > (lo + 0.4 * (hi - lo))), aud)
    thr = np.interp(np.arange(len(mf)), centers, levels)
    marg = np.interp(np.arange(len(mf)), centers, margins)
    # 3) hysteresis (Schmitt) around the tracking threshold
    hi_t = thr + 0.20 * marg
    lo_t = thr - 0.20 * marg
    on = np.empty(len(mf), bool)
    state = mf[0] > thr[0]
    for i in range(len(mf)):
        if state and mf[i] < lo_t[i]:
            state = False
        elif not state and mf[i] > hi_t[i]:
            state = True
        on[i] = state
    txt, info = _runs_to_text(_rle(on), aud)
    info["mf"] = True
    return txt, info


def _eye_and_fade(env):
    """Cheap eye-opening Q + fade depth (dB) straight from the envelope, so the
    router can pick a decoder without the full cw_quality apparatus."""
    e = env[env > 0].astype(np.float64)
    if len(e) < 200:
        return 0.0, 0.0
    thr = np.percentile(e, 55)
    on, off = e[e > thr], e[e <= thr]
    if len(on) < 20 or len(off) < 20:
        return 0.0, 0.0
    Q = (on.mean() - off.mean()) / (on.std() + off.std() + 1e-9)
    # fade depth from per-ON-run mark levels
    runs = _rle(env > thr)
    idx = 0; marks = []
    for s, ln in runs:
        if s and ln > 2:
            marks.append(float(np.median(env[idx:idx + ln])))
        idx += ln
    fade = 0.0
    if len(marks) >= 6:
        m = np.array(marks)
        fade = 20 * np.log10((np.percentile(m, 90) + 1e-9) / (np.percentile(m, 10) + 1e-9))
    return float(Q), float(fade)


def decode_env_auto(env, aud):
    """Apparatus-routed decode: use the classic threshold decoder when the eye is
    open (it's proven best on clean signals), and switch to the matched-filter +
    fade-tracking decoder only when the signal is FADING with a closing eye -
    exactly the 'loud but eye closed' case where the global threshold fails.
    Best-of-both with no regression on clean copy."""
    Q, fade = _eye_and_fade(env)
    if Q >= 4.5 and fade <= 4.0:            # pristine open eye -> classic (proven best on clean)
        txt, info = decode_env(env, aud)
        info["route"] = "classic"
        return txt, info
    # weak / fading / eye-closing / dit-collapsing -> matched filter + robust dit.
    # Campaign 2 (lab/science_log.md): this recovers copy where the global
    # threshold collapses to stuck-'T' garbage ('KEEP DOING WHAT' vs 'T T T T'),
    # and nearly doubles the synthetic copy-floor (0.475 -> 0.947), with the clean
    # open-eye case still routed to classic above so KI4XH-grade copy is untouched.
    txt, info = decode_env_mf2(env, aud)    # resolved at call time (defined below)
    info["route"] = "mf2"
    return txt, info


# ---- EXPERIMENTAL (Campaign 2: "copy the weakest Morse") ------------------
# Opt-in variants benched in weak_bench.py. Production (decode_env_auto) stays
# unchanged until a variant WINS the bench (SNR floor >= baseline AND fade floor
# up AND real recall == 13/13 AND test_cw.py green). See lab/science_log.md.

def _robust_dit(on_runs, aud):
    """Dit length that resists collapse to noise-chatter. Drop sub-8 ms runs (no
    real dit is that short at <=50 wpm) before clustering the ON runs, then take
    the shorter cluster; floor at aud*0.015 (~50 wpm) so a chattery envelope
    can't drive dit to ~2 samples (the wpm=4800 collapse artifact)."""
    o = np.array([r for r in on_runs if r >= aud * 0.008], float)   # 8 ms chatter floor
    if len(o) < 3:
        o = np.array(on_runs, float)
    if len(o) == 0:
        return aud * 0.06
    med = np.median(o)
    dit = np.median(o[o <= med]) or med
    return float(np.clip(dit, aud * 0.015, aud * 0.30))


def _mf_slice_decode(env, aud, ndit, lo, hi, rt=None):
    """Shared matched-filter back end: integrate over ndit, fade-track the slice
    level per ~0.6s block with a gentle noise gate + hysteresis, RLE -> text.
    Used by decode_env_mf2 (single dit estimate) and decode_env_mf3 (dit search).
    rt = run-length->text function (default _runs_to_text; _runs_to_text2 for H5
    graceful '?' degradation)."""
    rt = rt or _runs_to_text
    mf = np.convolve(env, np.ones(ndit, np.float32) / ndit, mode="same")
    blk = max(int(0.6 * aud), 4 * ndit)
    gfloor = np.percentile(mf, 60)
    centers, levels, margins = [], [], []
    for b in range(0, len(mf), blk):
        seg = mf[b:b + blk]
        if len(seg) < ndit:
            continue
        blo, bhi = np.percentile(seg, 25), np.percentile(seg, 92)
        centers.append(b + len(seg) / 2)
        if bhi < gfloor and bhi < 1.35 * blo:          # BOTH noise-floor AND flat -> suppress
            levels.append(bhi * 5 + 1e-6); margins.append(0.0)
        else:                                          # present (even if weak) -> slice lower
            levels.append(blo + 0.45 * (bhi - blo)); margins.append(0.45 * (bhi - blo))
    if len(centers) < 2:
        return rt(_rle(mf > (lo + 0.4 * (hi - lo))), aud), mf
    thr = np.interp(np.arange(len(mf)), centers, levels)
    marg = np.interp(np.arange(len(mf)), centers, margins)
    hi_t = thr + 0.20 * marg
    lo_t = thr - 0.20 * marg
    on = np.empty(len(mf), bool)
    state = mf[0] > thr[0]
    for i in range(len(mf)):
        if state and mf[i] < lo_t[i]:
            state = False
        elif not state and mf[i] > hi_t[i]:
            state = True
        on[i] = state
    return rt(_rle(on), aud), mf


def decode_env_mf2q(env, aud):
    """EXP H5: decode_env_mf2 with the graceful '?' back end (_runs_to_text2) -
    ambiguous dit/dah calls degrade to '?' instead of a confident wrong letter."""
    if len(env) < aud // 2:
        return decode_env(env, aud)
    hi, lo = np.percentile(env, 90), np.percentile(env, 25)
    if hi - lo < 1e-6:
        return "", {}
    on0 = env > (lo + 0.4 * (hi - lo))
    on0_runs = [ln for s, ln in _rle(on0) if s]
    if len(on0_runs) < 3:
        return "", {"runs": len(on0_runs)}
    ndit = int(np.clip(_robust_dit(on0_runs, aud), aud * 0.015, aud * 0.3))
    (txt, info), _ = _mf_slice_decode(env, aud, ndit, lo, hi, rt=_runs_to_text2)
    info["mf"] = "2q"
    return txt, info


def decode_env_mf2(env, aud):
    """EXP: matched-filter decoder with a robust dit and a GENTLER noise gate.

    Two changes vs decode_env_mf, targeting its two baseline failures:
      * robust dit (sizes the matched filter) so a chattery/fast envelope can't
        collapse the width to noise-chatter;
      * the per-block noise gate only fully suppresses a block that is BOTH below
        the global signal floor AND flat (bhi < 1.35*blo). decode_env_mf killed
        any block with bhi<1.6*blo, so on a globally low-eye fading real capture
        it suppressed real signal blocks and returned EMPTY (0/13 real recall).
    """
    if len(env) < aud // 2:
        return decode_env(env, aud)
    hi, lo = np.percentile(env, 90), np.percentile(env, 25)
    if hi - lo < 1e-6:
        return "", {}
    on0 = env > (lo + 0.4 * (hi - lo))
    on0_runs = [ln for s, ln in _rle(on0) if s]
    if len(on0_runs) < 3:
        return "", {"runs": len(on0_runs)}
    ndit = int(np.clip(_robust_dit(on0_runs, aud), aud * 0.015, aud * 0.3))
    (txt, info), _ = _mf_slice_decode(env, aud, ndit, lo, hi)
    info["mf"] = 2
    return txt, info


def decode_env_mf3(env, aud):
    """EXP H4 (NEGATIVE RESULT - NOT routed into production; kept opt-in so the
    loop doesn't re-try the same idea). Matched-filter with a per-window DIT-WIDTH
    SEARCH. Two selection criteria both FAILED the gate: eye-max improved the synth
    copy-floor (1.09 vs mf2's 0.947) but over-smoothed and garbled a real fast fist
    ('KEEP DOING WHAT' -> 'UEEP DDTND'); readability-scoring destroyed CLEAN copy
    (KI4XH -> 'MI TKT TTTKTTT', clean CER 0.26). LESSON: a per-window width search
    is unstable - one GLOBAL robust-dit estimate (mf2) is better-behaved across
    clean+weak+fading. The frontier is a better DETECTOR/timing model, not width
    re-selection. See lab/science_log.md EXP-2."""
    if len(env) < aud // 2:
        return decode_env(env, aud)
    hi, lo = np.percentile(env, 90), np.percentile(env, 25)
    if hi - lo < 1e-6:
        return "", {}
    on0 = env > (lo + 0.4 * (hi - lo))
    on0_runs = [ln for s, ln in _rle(on0) if s]
    if len(on0_runs) < 3:
        return "", {"runs": len(on0_runs)}
    d0 = _robust_dit(on0_runs, aud)
    # score each candidate width by decoded READABILITY (real letters minus '?'
    # penalty), NOT eye-opening: maximizing eye over-smooths and garbles fast real
    # fists ('KEEP DOING WHAT' -> 'UEEP DDTND'). Tie-break toward the NARROWER
    # width (less smoothing preserves fast elements).
    best = None
    for mult in (0.6, 0.8, 1.0, 1.25, 1.5):
        ndit = int(np.clip(d0 * mult, aud * 0.015, aud * 0.3))
        (txt, info), _ = _mf_slice_decode(env, aud, ndit, lo, hi)
        letters = sum(1 for c in txt if c.isalnum())
        score = letters - 2 * txt.count("?")            # readable copy, penalize junk
        if best is None or score > best[0] + 1e-9:      # strict > keeps earlier (narrower) on tie
            best = (score, ndit, txt, info)
    _, best_ndit, txt, info = best
    info["mf"] = 3
    info["ndit"] = best_ndit
    return txt, info


def decode_env_auto2(env, aud):
    """EXP H8 (WASH - NOT routed into production; kept opt-in). 3-way router adding
    the HSMM+Viterbi decoder for the weak-but-not-fading case. bayes has the best
    weak-SNR copy-floor (1.032 vs mf2 0.947, near-zero CER 0.18-0.75) but CANNOT
    handle fade (0.10); mf2 owns fade (0.443). Ensembling them (classic on pristine
    eye; mf2 when fading; else keep the more-readable of mf2/bayes with a garbage-
    volume guard) nets SNR 0.972 / FADE 0.436 - both WITHIN bench sampling noise of
    mf2, at 2x compute (bayes EM+Viterbi is slow for the live panel). VERDICT: not
    worth promoting as an ensemble. The real lever is bayes's weak-SNR strength;
    NEXT (H8b) = give the Viterbi front-end proper FADE TRACKING so one fast
    decoder gets BOTH the 1.03 SNR floor AND >=0.443 fade, then route it directly.
    See lab/science_log.md EXP-3."""
    Q, fade = _eye_and_fade(env)
    if Q >= 4.5 and fade <= 4.0:
        txt, info = decode_env(env, aud)
        info["route"] = "classic"
        return txt, info
    txt_m, info_m = decode_env_mf2(env, aud)
    if fade > 2.0:                              # ANY real fade -> the fade-tracker wins
        info_m["route"] = "mf2"                 # (bayes garbles QSB; threshold set between
        return txt_m, info_m                    #  no-fade ~0.6dB and QSB-0.6 ~3.7dB synth)
    try:
        import cw_bayes
        txt_b, info_b = cw_bayes.decode_bayes(env, aud, soft=False)
    except Exception:
        info_m["route"] = "mf2"
        return txt_m, info_m

    def _rd(t):
        return sum(c.isalnum() for c in t) - 3 * t.count("?")
    # prefer bayes ONLY if it's more readable AND not a garbage-volume blowup
    # (a mis-routed fade case garbles into long junk that would win on raw count).
    if _rd(txt_b) > _rd(txt_m) and len(txt_b) <= 1.4 * len(txt_m) + 6:
        info_b["route"] = "bayes"
        return txt_b, info_b
    info_m["route"] = "mf2"
    return txt_m, info_m


def find_offset(iq, fs, search=15000):
    N = 1 << 15
    seg = iq[:len(iq) // N * N].reshape(-1, N) * np.hanning(N).astype(np.float32)
    P = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2).mean(axis=0)
    c = N // 2
    k = int(search / (fs / N))
    band = P[c - k:c + k].copy()
    return (int(np.argmax(band)) - k) * fs / N



def aim(iq, fs, search=15000.0, prefer_cw=True):
    """Carrier aim for the LIVE paths. Defaults to the cocktail-party
    selector (cw_center: census + eye x rhythm audition), which found the
    true CW carrier a median 3.3 kHz from power-argmax across the whole
    harvested corpus (8/03). Falls back to find_offset if the selector
    finds no keyed candidate, so behaviour is never worse than before."""
    if prefer_cw:
        try:
            import cw_center
            res = cw_center.pick(iq, fs, search=search)
            b = res.get("best")
            if b and b.get("rhythm", 0) > 0:
                return float(b["hz"]), b
        except Exception:
            pass
    return float(find_offset(iq, fs, search)), None


def cmd_selftest(args):
    print("=" * 60)
    print("hamTuna CW self-test (synthesize -> noise -> decode)")
    print("=" * 60)
    fs = 250000.0
    aud = fs
    msg = "NDB"
    dit = int(0.06 * fs)                    # ~20 wpm
    seq = []
    for i, ch in enumerate(msg):
        for el in INV[ch]:
            seq.append((1, dit if el == "." else 3 * dit))
            seq.append((0, dit))            # intra-char gap
        seq.append((0, 3 * dit))            # letter gap
    sig = []
    for s, ln in seq:
        sig.append(np.full(ln, float(s)))
    key = np.concatenate(sig)
    t = np.arange(len(key))
    iq = (key * np.exp(2j * np.pi * -6200 / fs * t)).astype(np.complex64)
    rng = np.random.default_rng(1)
    iq += (rng.normal(0, 0.15, len(iq)) + 1j * rng.normal(0, 0.15, len(iq))).astype(np.complex64)
    env, a = envelope(iq, fs, -6200)
    txt, info = decode_env(env, a)
    ok = "NDB" in txt.replace(" ", "")
    print(f"  sent 'NDB' -> decoded '{txt}'  {info}")
    print(f"  {'OK' if ok else 'FAIL'}")
    print("=" * 60)
    print("SELFTEST", "PASS" if ok else "FAIL")
    print("=" * 60)
    return 0 if ok else 1


def cmd_decode(args):
    raw = np.fromfile(args.file, dtype=np.int16).astype(np.float32) / 32768.0
    iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    off = args.offset
    if off is None:
        off, pick = aim(iq, args.fs)
        if pick:
            print(f"[cw] auto-centered on {off:+.0f} Hz "
                  f"(eye {pick['eye']}, rhythm {pick['rhythm']}, "
                  f"{pick['wpm']:.0f} wpm)")
        else:
            print(f"[cw] no keyed carrier found - power-argmax "
                  f"{off:+.0f} Hz")
    env, a = envelope(iq, args.fs, off)
    txt, info = decode_env(env, a)
    wpm = info.get("wpm", 0)
    print(f"[cw] {info}")
    if not (3 <= wpm <= 45):
        print("[cw] NO READABLE CW - keying rate out of range "
              "(noise, weak signal, or A2A tone-keyed beacon). "
              "Try a longer/stronger capture.")
        return ""
    print(f"[cw] decoded: '{txt}'")
    if txt and txt.replace(" ", "").isalnum():
        print("  -> looks like a real ID! (NDB IDs are 1-3 letters, repeated)")
    return txt


def _ensure_sdr_dll_path():
    """Windows + conda-style python: SoapySDR driver DLLs aren't on PATH
    unless the environment is activated - fix it here so bare launches work."""
    import os
    if os.name != "nt":
        return
    root = Path(sys.executable).resolve().parent
    for p in (root / "Library" / "bin",
              Path(r"C:\Program Files\SDRplay\API\x64"),
              Path(r"C:\Program Files\SDRplay\API")):
        if p.is_dir():
            os.environ["PATH"] = str(p) + os.pathsep + os.environ["PATH"]
            try:
                os.add_dll_directory(str(p))
            except Exception:
                pass


def _open_sdr(antenna, fs=250_000.0):
    # rate-ok: magnitude-only HF use, verified working daily (CW harvester);
    # phase-sensitive callers must pass >=2048000 (aprs_rx/pager_rx already do)
    _ensure_sdr_dll_path()
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16
    SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
    sdr = SoapySDR.Device("driver=sdrplay")
    sdr.setSampleRate(SOAPY_SDR_RX, 0, fs)
    try:
        sdr.setAntenna(SOAPY_SDR_RX, 0, antenna)
    except Exception:
        pass
    try:
        sdr.setGainMode(SOAPY_SDR_RX, 0, False)
        sdr.setGain(SOAPY_SDR_RX, 0, "IFGR", 30)
        sdr.writeSetting("rfgain_sel", "0")
    except Exception:
        pass
    st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16)
    sdr.activateStream(st)
    return sdr, st


def _grab(sdr, st, secs, fs=250_000.0, max_stall_s=None):
    """Read secs of IQ. If max_stall_s is set, raise once that long passes
    with NO samples delivered.

    Why the stall guard exists (2026-08-01): a wedged RSPdx can OPEN cleanly
    and then deliver nothing — readStream just times out forever with
    ret == -1, which the loop below treats as 'keep waiting'. The meteor
    baseline spun inside this loop for 3 hours on a silent stream, its
    3-hour deadline unreachable because it was checked between grabs that
    never returned. Default None keeps the historical behavior for the
    harvester and every other caller; long-running unattended consumers
    should pass a bound, because a stream that says nothing for a minute is
    not late — it is dead, and the honest move is to crash loudly.
    """
    n_want = int(secs * fs)
    buf = np.empty(2 * 65536, np.int16)
    out = np.empty(2 * n_want, np.int16)
    got = 0
    t_last = time.time()
    while got < n_want:
        r = sdr.readStream(st, [buf], 65536, timeoutUs=1_000_000)
        if r.ret > 0:
            n = min(r.ret, n_want - got)
            out[2 * got:2 * (got + n)] = buf[:2 * n]
            got += n
            t_last = time.time()
        elif r.ret < 0 and r.ret != -1:
            break
        if max_stall_s and time.time() - t_last > max_stall_s:
            raise RuntimeError(
                f"SDR stream stalled: no samples for {max_stall_s:.0f}s "
                f"({got}/{n_want} delivered). Device opens but does not "
                f"stream — restart SDRplayAPIService and run a sacrificial "
                f"stream probe; an OPEN is not a health check.")
    return ((out[0::2].astype(np.float32) + 1j * out[1::2].astype(np.float32))
            / 32768.0).astype(np.complex64)[:got]


def cmd_listen(args):
    """Live capture on a CW-active frequency, then decode. Real reads are
    logged to lab/cw_decodes.jsonl."""
    import json
    import time as _t
    from SoapySDR import SOAPY_SDR_RX
    sdr, st = _open_sdr(args.antenna, args.fs)
    sdr.setFrequency(SOAPY_SDR_RX, 0, args.khz * 1e3)
    _t.sleep(0.2)
    iq = _grab(sdr, st, args.secs, args.fs, max_stall_s=60)
    sdr.deactivateStream(st); sdr.closeStream(st)
    off, pick = aim(iq, args.fs)
    if pick:
        print(f"[cw] auto-centered {off:+.0f} Hz (rhythm {pick['rhythm']}, "
              f"{pick['wpm']:.0f} wpm)", flush=True)
    env, a = envelope(iq, args.fs, off)
    txt, info = decode_env(env, a)
    wpm = info.get("wpm", 0)
    if 3 <= wpm <= 45 and txt.strip():
        print(f"[cw] {args.khz} kHz  {info}")
        print(f"[cw] MORSE DECODED: '{txt}'")
        lab = Path(__file__).resolve().parent.parent / "lab"
        lab.mkdir(exist_ok=True)
        rec = {"ts": _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime()),
               "khz": args.khz, "wpm": wpm, "text": txt}
        with open(lab / "cw_decodes.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    else:
        print(f"[cw] {args.khz} kHz: no readable CW (wpm {wpm})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    d = sub.add_parser("decode")
    d.add_argument("--file", required=True)
    d.add_argument("--offset", type=float, default=None)
    d.add_argument("--fs", type=float, default=250000)
    li = sub.add_parser("listen")
    li.add_argument("--khz", type=float, default=14030)   # 20m CW calling area
    li.add_argument("--secs", type=float, default=30)
    li.add_argument("--antenna", default="Antenna C")
    li.add_argument("--fs", type=float, default=250000)
    args = ap.parse_args()
    if args.cmd == "selftest":
        sys.exit(cmd_selftest(args))
    elif args.cmd == "decode":
        cmd_decode(args)
    else:
        cmd_listen(args)


if __name__ == "__main__":
    main()
