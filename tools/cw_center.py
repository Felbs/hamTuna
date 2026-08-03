"""cw_center.py - the cocktail-party lock: pick the RIGHT CW carrier.

FABLE5 queue item #1. `find_offset` picks the power-argmax carrier,
which in a pileup is whoever is loudest - not whoever you are copying.
EXP-7 showed a centered narrow filter buys ~6x QRM rejection, but only
when it is centered on the intended signal; centered on the wrong one
it rejects the signal you wanted. So the selector must rank carriers by
COPYABILITY, not power.

The method:
  1. CENSUS - average periodogram over the whole capture, find every
     peak that clears the noise floor (not just the maximum), dedup to
     one candidate per ~80 Hz so a single carrier's skirts do not
     register as several.
  2. AUDITION - demodulate each candidate through the narrow envelope
     path and score it on what a human ear actually uses:
       * eye-opening Q (the CW "MER" - mark/space separation),
       * KEYING RHYTHM: element durations must cluster into a dit/dah
         pair with a plausible ratio (2-4x) and a plausible dit length
         (20-200 ms => 6-60 wpm). Noise and carriers have no rhythm;
         this is what separates a CW op from a heterodyne or a birdie.
     score = eye * rhythm_confidence.
  3. HYSTERESIS - in `Tracker`, a challenger must beat the held carrier
     by a margin for N consecutive frames before the lock moves. A
     tracker that flips to whoever is momentarily loudest is exactly the
     failure mode we are fixing.

  python cw_center.py bench            # corpus A/B vs power-argmax
  python cw_center.py show FILE.cs16   # candidate table for one capture
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cw  # noqa: E402

HARVEST = HERE.parent / "lab" / "cw_harvest"

DEDUP_HZ = 80.0            # one candidate per this span
FLOOR_DB = 6.0             # peak must clear the median floor by this
MAX_CAND = 12              # audition budget per capture
DIT_MS = (20.0, 200.0)     # 60 wpm .. 6 wpm
DAH_RATIO = (1.8, 4.5)     # textbook 3; real ops and QSB drift wider


def census(iq, fs, search=15000.0, nfft=1 << 15):
    """Every carrier-like peak in +-search, strongest first: [(hz, db)]."""
    seg = iq[:len(iq) // nfft * nfft]
    if len(seg) < nfft:
        return []
    seg = seg.reshape(-1, nfft) * np.hanning(nfft).astype(np.float32)
    P = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2
         ).mean(axis=0)
    c = nfft // 2
    k = int(search / (fs / nfft))
    band = P[c - k:c + k]
    db = 10 * np.log10(band + 1e-20)
    floor = float(np.median(db))
    bins_per = max(1, int(DEDUP_HZ / (fs / nfft)))
    out = []
    work = db.copy()
    for _ in range(MAX_CAND):
        i = int(np.argmax(work))
        if work[i] < floor + FLOOR_DB:
            break
        out.append(((i - k) * fs / nfft, float(db[i] - floor)))
        work[max(0, i - bins_per):i + bins_per] = -999.0
    return out


def key_runs(env, aud, smooth_ms=8.0):
    """ON/OFF runs of the KEYING, not of the noise. Two fixes the naive
    slicer needs (measured 8/03: raw slicing gave 3-15 ms 'dits' at 45%
    duty on a strong capture - it was chopping envelope noise):
      * smooth with a ~8 ms moving average (well under a 20 wpm 60 ms
        dit, well over the noise correlation time),
      * SCHMITT trigger between the off/on cluster levels so a mark does
        not shatter into fragments every time it dips.
    """
    n = max(1, int(smooth_ms * 1e-3 * aud))
    sm = np.convolve(env.astype(np.float64), np.ones(n) / n, "same")
    lo_lvl = float(np.percentile(sm, 30))
    hi_lvl = float(np.percentile(sm, 92))
    if hi_lvl <= lo_lvl:
        return []
    hi = lo_lvl + 0.65 * (hi_lvl - lo_lvl)
    lo = lo_lvl + 0.35 * (hi_lvl - lo_lvl)
    state = sm[0] > hi
    runs = []
    start = 0
    for i in range(1, len(sm)):
        if state and sm[i] < lo:
            runs.append((1, i - start)); start = i; state = False
        elif not state and sm[i] > hi:
            runs.append((0, i - start)); start = i; state = True
    runs.append((1 if state else 0, len(sm) - start))
    return runs


def rhythm_score(env, aud):
    """Confidence that this envelope is HAND-KEYED CW, from element
    durations alone: ON runs should cluster into dit/dah with a sane
    ratio and a sane dit length. Returns (0..1, wpm_estimate)."""
    runs = key_runs(env, aud)
    ons = [ln for s, ln in runs if s and ln > 1]
    if len(ons) < 8:
        return 0.0, 0.0
    d = np.array(ons, float) / aud * 1000.0        # ms
    # dit/dah by 2-means on log durations, NOT fixed percentiles: a
    # sending with more dits than dahs pushes p85 down into the dit
    # cluster and the ratio test then rejects real CW (measured 8/03 on
    # an 18 wpm capture that scored 1.75 instead of ~3).
    lg = np.log(np.clip(d, 1e-3, None))
    lo, hi = float(lg.min()), float(lg.max())
    if hi - lo < 0.15:                 # single cluster: no dit/dah pair
        return 0.0, 0.0
    c1, c2 = lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)
    for _ in range(25):
        m = np.abs(lg - c1) <= np.abs(lg - c2)
        if not m.any() or m.all():
            return 0.0, 0.0
        c1, c2 = float(lg[m].mean()), float(lg[~m].mean())
    dit, dah = float(np.exp(c1)), float(np.exp(c2))
    if not (DIT_MS[0] <= dit <= DIT_MS[1]):
        return 0.0, 0.0
    ratio = dah / max(dit, 1e-6)
    if not (DAH_RATIO[0] <= ratio <= DAH_RATIO[1]):
        return 0.0, 0.0
    # tightness of the two clusters = how machine-like the keying is
    spread = float(np.mean(np.minimum(np.abs(lg - c1), np.abs(lg - c2))))
    tight = max(0.0, 1.0 - spread / 0.35)
    ratio_fit = 1.0 - min(abs(ratio - 3.0) / 1.6, 1.0)
    return float(tight * (0.55 + 0.45 * ratio_fit)), 1200.0 / dit


def audition(iq, fs, hz, bw_hz=150.0, aud=8000.0):
    """Score one candidate carrier: (score, eye, rhythm, wpm)."""
    env, a = cw.envelope2(iq, fs, hz, aud=aud, bw_hz=bw_hz)
    eye, _fade = cw._eye_and_fade(env)
    rh, wpm = rhythm_score(env, a)
    return eye * rh, eye, rh, wpm


def pick(iq, fs, search=15000.0, bw_hz=150.0, verbose=False):
    """The selector: audition every candidate, return the best-copy one.
    Returns dict(hz, score, eye, rhythm, wpm, candidates)."""
    cands = census(iq, fs, search)
    scored = []
    for hz, snr_db in cands:
        s, eye, rh, wpm = audition(iq, fs, hz, bw_hz)
        scored.append({"hz": round(hz, 1), "snr_db": round(snr_db, 1),
                       "score": round(s, 3), "eye": round(eye, 2),
                       "rhythm": round(rh, 2), "wpm": round(wpm)})
    scored.sort(key=lambda d: -d["score"])
    best = scored[0] if scored else None
    if verbose:
        print(f"{'Hz':>9}{'SNRdB':>7}{'eye':>7}{'rhythm':>8}"
              f"{'wpm':>6}{'score':>8}")
        for d in scored:
            print(f"{d['hz']:>9.1f}{d['snr_db']:>7.1f}{d['eye']:>7.2f}"
                  f"{d['rhythm']:>8.2f}{d['wpm']:>6.0f}{d['score']:>8.3f}")
    return {"best": best, "candidates": scored}


class Tracker:
    """Live carrier lock with hysteresis: hold the current signal unless
    a challenger beats it by `margin` for `patience` consecutive frames.
    The whole point of the queue item - a lock that does not flap when
    someone louder keys up next door."""

    def __init__(self, margin=1.25, patience=3, max_drift_hz=120.0):
        self.margin = margin
        self.patience = patience
        self.max_drift_hz = max_drift_hz
        self.locked_hz = None
        self.locked_score = 0.0
        self._challenger = None
        self._votes = 0

    def update(self, iq, fs, search=15000.0, bw_hz=150.0):
        res = pick(iq, fs, search, bw_hz)
        cands = res["candidates"]
        if not cands:
            return self.locked_hz
        best = cands[0]
        if self.locked_hz is None:
            self.locked_hz, self.locked_score = best["hz"], best["score"]
            return self.locked_hz
        # follow small drift of the SAME signal (op's VFO, doppler, chirp)
        near = [c for c in cands
                if abs(c["hz"] - self.locked_hz) <= self.max_drift_hz]
        if near:
            held = max(near, key=lambda c: c["score"])
            self.locked_hz = held["hz"]        # track the drift
            self.locked_score = held["score"]
        else:
            held = {"score": 0.0}
        if best["hz"] == self.locked_hz:
            self._challenger, self._votes = None, 0
            return self.locked_hz
        # a different carrier is winning - make it earn the switch
        if best["score"] > self.margin * max(held["score"], 1e-6):
            if self._challenger is not None \
                    and abs(best["hz"] - self._challenger) <= self.max_drift_hz:
                self._votes += 1
            else:
                self._challenger, self._votes = best["hz"], 1
            if self._votes >= self.patience:
                self.locked_hz = best["hz"]
                self.locked_score = best["score"]
                self._challenger, self._votes = None, 0
        else:
            self._challenger, self._votes = None, 0
        return self.locked_hz


# ==========================================================================
# corpus bench: does eye+rhythm selection beat power-argmax?
# ==========================================================================
def _load(path, n_max=None):
    raw = np.fromfile(path, np.int16, count=(2 * n_max) if n_max else -1)
    return ((raw[0::2].astype(np.float32)
             + 1j * raw[1::2].astype(np.float32)) / 32768.0
            ).astype(np.complex64)


def cmd_bench(args):
    js = sorted(glob.glob(str(HARVEST / "cw_*.json")))
    if args.limit:
        js = js[:args.limit]
    fs = 250000.0
    wins = losses = ties = 0
    d_eye = []
    print(f"{'capture':<34}{'argmax':>9}{'picked':>10}"
          f"{'eyeA':>7}{'eyeP':>7}")
    for j in js:
        d = json.loads(Path(j).read_text())
        iqp = HARVEST / d.get("iq_file", "")
        if not iqp.is_file():
            continue
        iq = _load(iqp, n_max=int(6 * fs))
        if len(iq) < int(2 * fs):
            continue
        off_argmax = cw.find_offset(iq, fs)
        _s, eye_a, _r, _w = audition(iq, fs, off_argmax)
        res = pick(iq, fs)
        if not res["best"]:
            continue
        off_pick = res["best"]["hz"]
        eye_p = res["best"]["eye"]
        d_eye.append(eye_p - eye_a)
        if eye_p > eye_a + 0.05:
            wins += 1
        elif eye_a > eye_p + 0.05:
            losses += 1
        else:
            ties += 1
        print(f"{Path(j).stem:<34}{off_argmax:>9.0f}{off_pick:>10.0f}"
              f"{eye_a:>7.2f}{eye_p:>7.2f}")
    n = wins + losses + ties
    if n:
        print(f"\nBENCH over {n} captures: selector WINS {wins}, "
              f"loses {losses}, ties {ties}")
        print(f"mean eye delta {np.mean(d_eye):+.3f} "
              f"(median {np.median(d_eye):+.3f})")
        print("VERDICT:", "selector beats power-argmax"
              if wins > losses else "no improvement - investigate")
    return 0


def cmd_show(args):
    fs = args.fs
    iq = _load(args.file, n_max=int(8 * fs))
    print(f"[show] {Path(args.file).name}  {len(iq)/fs:.1f}s @ {fs/1e3:.0f} kS/s")
    print(f"[show] power-argmax says {cw.find_offset(iq, fs):+.0f} Hz")
    res = pick(iq, fs, verbose=True)
    if res["best"]:
        b = res["best"]
        print(f"[show] SELECTED {b['hz']:+.0f} Hz  "
              f"(eye {b['eye']}, rhythm {b['rhythm']}, {b['wpm']:.0f} wpm)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--limit", type=int, default=60)
    s = sub.add_parser("show")
    s.add_argument("file")
    s.add_argument("--fs", type=float, default=250000.0)
    a = ap.parse_args()
    sys.exit(cmd_bench(a) if a.cmd == "bench" else cmd_show(a))


if __name__ == "__main__":
    main()
