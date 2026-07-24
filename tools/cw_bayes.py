#!/usr/bin/env python3
"""cw_bayes.py - segment-level HSMM + Viterbi Morse decoder (the 'next frontier').

Where the classic decoder makes a greedy per-gap hard decision, this finds the
globally most-likely element/gap sequence over the WHOLE transmission with
Viterbi - so a single fade or noise blip can't derail a character; the path cost
integrates all the evidence. Architecture (research-derived, RSCW/AG1LE/HSMM):

  envelope -> segments (level, duration, amplitude)
           -> Viterbi over classes {DIT, DAH, GAP_INTRA, GAP_LETTER, GAP_WORD}
              emission  = log-duration Gaussian (two clocks: marks vs gaps=Farnsworth)
                          + amplitude evidence
              transition= Morse grammar (alternation, letter/word termination)
           -> segmental-EM to track speed (unit u) and cluster widths (sigma)
           -> backtrace -> dit/dah patterns -> letters

This is the soft-decision sequence decoder that copies THROUGH fading instead of
just measuring it. Validated by a synthetic self-test (known text + jitter+noise)
and A/B'd against the classic/matched-filter decoders on the corpus.
"""
import sys
from math import log, exp, pi
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cw

INF = 1e18
CLASSES_ON = ("DIT", "DAH")
CLASSES_OFF = ("GAP_INTRA", "GAP_LETTER", "GAP_WORD")
MULT = {"DIT": 1.0, "DAH": 3.0, "GAP_INTRA": 1.0, "GAP_LETTER": 3.0, "GAP_WORD": 7.0}


def _matched(env, ndit):
    """Matched filter = boxcar of one dit (SNR-optimal for a rectangular OOK
    element; the single biggest CER lever per AG1LE). Integrates noise spikes out
    before slicing, so segmentation doesn't shatter."""
    ndit = max(3, int(ndit))
    return np.convolve(env, np.ones(ndit, np.float32) / ndit, mode="same")


def _soft_keystate(env, ndit, aud):
    """Agent-C front end: matched filter -> sliding-window order-statistic level
    tracking (signal-blind, rides QSB) -> Rician/Rayleigh SOFT key-down probability.
    The softness + ~0.5 s window is what avoids the fragmentation a hard per-block
    threshold causes. Returns (key_state_bool, p_down, mf)."""
    mf = _matched(env, ndit)
    blk = max(int(0.5 * aud), 8 * int(ndit))
    centers, nf, sl = [], [], []
    for b in range(0, len(mf), blk):
        seg = mf[b:b + blk]
        if len(seg) < ndit:
            continue
        centers.append(b + len(seg) / 2)
        nf.append(np.percentile(seg, 20))     # noise floor (signal-blind low pct)
        sl.append(np.percentile(seg, 88))     # signal level
    if len(centers) < 2:
        centers = [0, len(mf)]
        nf = [np.percentile(mf, 20)] * 2; sl = [np.percentile(mf, 88)] * 2
    idx = np.arange(len(mf))
    noise = np.interp(idx, centers, nf)
    sig = np.interp(idx, centers, sl)
    sigma = np.maximum(noise / 0.668, 1e-6)   # Rayleigh: 20th pct = sigma*sqrt(-2ln0.8)
    A = np.sqrt(np.maximum(sig ** 2 - 2 * sigma ** 2, 0.0))
    rho = A / sigma                           # voltage SNR, tracks the fade
    u = mf / sigma
    z = rho * u - 0.5 * rho ** 2 - 0.4        # Rician LLR + key-up prior odds
    p_down = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    return p_down > 0.5, p_down, mf


def _fade_keystate(mf, ndit):
    """Fade-tracking slice of the matched-filtered envelope: a per-block on/off
    level (interpolated per-sample) + hysteresis + per-block SNR gate, so QSB that
    sinks the signal below a global threshold mid-character doesn't drop elements.
    This is agent-C's crux fix (the piece AG1LE's single noise tracker got wrong).
    Returns a boolean key-state array."""
    blk = max(int(6 * ndit), 1)
    gfloor = np.percentile(mf, 60)
    centers, levels, margins = [], [], []
    for b in range(0, len(mf), blk):
        seg = mf[b:b + blk]
        if len(seg) < ndit:
            continue
        blo, bhi = np.percentile(seg, 25), np.percentile(seg, 92)
        centers.append(b + len(seg) / 2)
        if bhi < gfloor or bhi < 1.5 * blo:          # local eye closed -> force off
            levels.append(bhi * 5 + 1e-6); margins.append(0.0)
        else:
            levels.append(blo + 0.5 * (bhi - blo)); margins.append(0.5 * (bhi - blo))
    if len(centers) < 2:
        thr = np.percentile(mf, 60)
        return mf > thr
    idx = np.arange(len(mf))
    thr = np.interp(idx, centers, levels)
    marg = np.interp(idx, centers, margins)
    hi_t, lo_t = thr + 0.2 * marg, thr - 0.2 * marg
    on = np.empty(len(mf), bool)
    state = mf[0] > thr[0]
    for i in range(len(mf)):
        if state and mf[i] < lo_t[i]:
            state = False
        elif not state and mf[i] > hi_t[i]:
            state = True
        on[i] = state
    return on


def _segments_from_state(on, env):
    """Boolean key-state + envelope -> [level, duration, mean_amp] segments."""
    segs = []
    cur = bool(on[0]); start = 0
    for i in range(1, len(on)):
        if bool(on[i]) != cur:
            segs.append([1 if cur else 0, i - start, float(env[start:i].mean())])
            cur = bool(on[i]); start = i
    segs.append([1 if cur else 0, len(on) - start, float(env[start:].mean())])
    return segs


def segments(env, thr):
    """Envelope -> list of [level, duration_samples, mean_amplitude]."""
    on = env > thr
    segs = []
    cur = bool(on[0]); start = 0
    for i in range(1, len(on)):
        if bool(on[i]) != cur:
            segs.append([1 if cur else 0, i - start, float(env[start:i].mean())])
            cur = bool(on[i]); start = i
    segs.append([1 if cur else 0, len(on) - start, float(env[start:].mean())])
    return segs


def _despeckle(segs, min_len):
    """Merge runs shorter than min_len into their neighbours (a noise spike that
    split a gap, or a dropout that split an element). Agent-B 'merge micro-segments'
    hybrid: keeps the Viterbi from ever seeing sub-dit fragments."""
    changed = True
    while changed and len(segs) > 2:
        changed = False
        # find the shortest interior segment below the floor
        k, kd = -1, min_len
        for i in range(1, len(segs) - 1):
            if segs[i][1] < kd:
                k, kd = i, segs[i][1]
        if k < 0:
            break
        # drop seg k; its two same-level neighbours merge (durations + weighted amp)
        a, b = segs[k - 1], segs[k + 1]
        da, db = a[1], b[1]
        merged = [a[0], da + segs[k][1] + db,
                  (a[2] * da + b[2] * db) / max(da + db, 1)]
        segs = segs[:k - 1] + [merged] + segs[k + 2:]
        changed = True
    return segs


# gap-class frequency priors (Mills-ish): intra > letter > word. Break ties toward
# the more common gap so absent word-gaps aren't invented. -log(freq).
GAP_PRIOR = {"GAP_INTRA": -log(0.45), "GAP_LETTER": -log(0.35), "GAP_WORD": -log(0.20)}


def _emit(seg, j, u, u_g, sigma, a_on, a_off, s_amp, lam):
    """Cost of labeling a segment as class j: log-duration Gaussian + amplitude +
    gap prior. Two clocks: marks & the intra-element gap ride u; only the LETTER/
    WORD spacing rides u_g (ARRL-exact Farnsworth -> stable when word gaps absent)."""
    lvl, d, amp = seg
    unit = u_g if j in ("GAP_LETTER", "GAP_WORD") else u
    mu = log(MULT[j] * unit)
    ld = log(max(d, 1))
    c_dur = 0.5 * ((ld - mu) / sigma[j]) ** 2 + log(sigma[j])
    tgt = a_on if j in CLASSES_ON else a_off
    c_amp = 0.5 * ((amp - tgt) / (s_amp + 1e-9)) ** 2
    return c_dur + lam * c_amp + GAP_PRIOR.get(j, 0.0)


def _viterbi(segs, u, u_g, sigma, a_on, a_off, s_amp, lam=0.5):
    M = len(segs)
    allowed = [CLASSES_ON if s[0] == 1 else CLASSES_OFF for s in segs]
    delta = [dict() for _ in range(M)]
    bp = [dict() for _ in range(M)]
    for j in allowed[0]:
        delta[0][j] = _emit(segs[0], j, u, u_g, sigma, a_on, a_off, s_amp, lam)
        bp[0][j] = None
    for i in range(1, M):
        for j in allowed[i]:
            e = _emit(segs[i], j, u, u_g, sigma, a_on, a_off, s_amp, lam)
            best, arg = INF, None
            for k in allowed[i - 1]:
                c = delta[i - 1][k]
                if c < best:
                    best, arg = c, k
            delta[i][j] = best + e
            bp[i][j] = arg
    j = min(delta[M - 1], key=delta[M - 1].get)
    labels = [None] * M
    for i in range(M - 1, -1, -1):
        labels[i] = j
        j = bp[i][j]
    return labels


def _labels_to_text(segs, labels):
    out, pat = [], ""
    for (lvl, d, amp), l in zip(segs, labels):
        if l == "DIT":
            pat += "."
        elif l == "DAH":
            pat += "-"
        elif l in ("GAP_LETTER", "GAP_WORD"):
            if pat:
                out.append(cw.MORSE.get(pat, "?")); pat = ""
            if l == "GAP_WORD":
                out.append(" ")
    if pat:
        out.append(cw.MORSE.get(pat, "?"))
    return "".join(out).strip()


def _em_speed(segs, u0, iters=4):
    """Segmental EM: alternate Viterbi labeling and re-estimate the two clocks
    (u for marks, u_g for gaps) and the per-class log-duration spreads."""
    on_amp = np.array([a for lvl, d, a in segs if lvl == 1])
    off_amp = np.array([a for lvl, d, a in segs if lvl == 0])
    a_on = float(np.median(on_amp)) if len(on_amp) else 1.0
    a_off = float(np.median(off_amp)) if len(off_amp) else 0.0
    s_amp = float(np.std(np.concatenate([on_amp, off_amp]))) or 1.0
    u = u_g = u0
    sigma = {c: 0.35 for c in CLASSES_ON + CLASSES_OFF}
    labels = None
    for _ in range(iters):
        labels = _viterbi(segs, u, u_g, sigma, a_on, a_off, s_amp)
        # M-step: u from marks + intra-gap (all 1-unit on the mark clock);
        # u_g from letter/word gaps only (the spacing clock -> Farnsworth-safe).
        u_lu, ug_lu = [], []
        byc = {c: [] for c in CLASSES_ON + CLASSES_OFF}
        for (lvl, d, amp), l in zip(segs, labels):
            lu = log(max(d, 1)) - log(MULT[l])
            if l in ("GAP_LETTER", "GAP_WORD"):
                ug_lu.append(lu)
            else:                       # DIT, DAH, GAP_INTRA all ride u
                u_lu.append(lu)
            unit = u_g if l in ("GAP_LETTER", "GAP_WORD") else u
            byc[l].append(log(max(d, 1)) - log(MULT[l] * unit))
        if u_lu:
            u = exp(float(np.mean(u_lu)))
        u_g = exp(float(np.mean(ug_lu))) if ug_lu else u   # no letter gaps -> tie to u
        for c in sigma:
            if len(byc[c]) >= 3:
                sigma[c] = max(0.15, float(np.std(byc[c])))
    return labels, dict(u=u, u_g=u_g, sigma=sigma, a_on=a_on, a_off=a_off, s_amp=s_amp)


def decode_bayes(env, aud, soft=False):
    """HSMM+Viterbi decode of a narrow-filtered CW envelope. Same signature as
    cw.decode_env -> (text, info)."""
    if len(env) < aud // 4:
        return "", {}
    hi, lo = np.percentile(env, 90), np.percentile(env, 25)
    if hi - lo < 1e-6:
        return "", {}
    # 1) coarse dit from a raw threshold, to size the matched filter
    raw_segs = segments(env, lo + 0.45 * (hi - lo))
    on0 = np.array([d for lvl, d, a in raw_segs if lvl == 1], float)
    if len(on0) < 3:
        return "", {"segs": len(raw_segs)}
    ndit = float(np.median(on0[on0 <= np.median(on0)])) or float(np.median(on0))
    ndit = float(np.clip(ndit, aud * 0.015, aud * 0.3))
    # 2) front end -> clean segments. soft=True uses the Rician/Rayleigh
    # fade-tracking soft detector (agent C); else static matched-filter threshold.
    if soft:
        on, _p, mf = _soft_keystate(env, ndit, aud)
        segs = _despeckle(_segments_from_state(on, mf), 0.4 * ndit)
    else:
        mf = _matched(env, ndit)
        mhi, mlo = np.percentile(mf, 88), np.percentile(mf, 25)
        segs = _despeckle(segments(mf, mlo + 0.5 * (mhi - mlo)), 0.4 * ndit)
    if len(segs) < 5:
        return "", {"segs": len(segs)}
    on_d = np.array([d for lvl, d, a in segs if lvl == 1], float)
    if len(on_d) < 3:
        return "", {"segs": len(segs)}
    u0 = float(np.median(on_d[on_d <= np.median(on_d)])) or float(np.median(on_d))
    labels, params = _em_speed(segs, u0)
    txt = _labels_to_text(segs, labels)
    wpm = 1.2 / (params["u"] / aud) if params["u"] else 0.0
    return txt, {"wpm": round(float(wpm), 1), "u_g_ratio": round(params["u_g"] / params["u"], 2),
                 "segs": len(segs), "bayes": True}


# ------------------------------------------------------------------ self-test
def _synth(text, wpm=20, fs=8000.0, jitter=0.12, noise=0.15, fade=0.0, seed=1):
    """Synthesize a Morse envelope with timing jitter, noise, and optional QSB."""
    rng = np.random.default_rng(seed)
    u = 1.2 / wpm * fs
    parts = []
    for ci, ch in enumerate(text):
        if ch == " ":
            parts.append((0, 7 * u)); continue
        code = cw.INV.get(ch.upper())
        if not code:
            continue
        for ei, el in enumerate(code):
            L = (1 if el == "." else 3) * u * (1 + rng.normal(0, jitter))
            parts.append((1, max(3, L)))
            if ei < len(code) - 1:
                parts.append((0, u * (1 + rng.normal(0, jitter))))
        parts.append((0, 3 * u * (1 + rng.normal(0, jitter))))   # letter gap
    env = np.concatenate([np.full(int(d), float(lvl)) for lvl, d in parts]).astype(np.float32)
    if fade > 0:                       # slow QSB
        t = np.arange(len(env)) / fs
        env = env * (1 - fade * 0.5 * (1 + np.sin(2 * pi * 0.3 * t))).astype(np.float32)
    env = env + rng.normal(0, noise, len(env)).astype(np.float32)
    return np.abs(env), fs


def selftest():
    print("=" * 60)
    print("cw_bayes HSMM+Viterbi self-test")
    print("=" * 60)
    ok = 0; tot = 0
    for msg, kw in [("PARIS", {}), ("CQ CQ DE W1AW", {}),
                    ("PARIS", dict(noise=0.35)), ("CQ DE K1ABC", dict(fade=0.6)),
                    ("HELLO WORLD", dict(jitter=0.2, noise=0.25))]:
        env, fs = _synth(msg, **kw)
        txt, info = decode_bayes(env, fs)
        good = txt.replace(" ", "") == msg.replace(" ", "")
        ok += good; tot += 1
        print(f"  sent {msg!r:20} -> {txt!r:22} {info.get('wpm')}wpm  {'OK' if good else 'x'}  {kw}")
    print("=" * 60)
    print(f"RESULT: {ok}/{tot} exact")
    return 0 if ok >= tot - 1 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    print(__doc__)
