#!/usr/bin/env python3
"""weak_bench.py - the CW decoder science bench: "how weak can we copy?"

You can't evolve a decoder toward the weakest signals without a reproducible
metric for HOW weak it copies. This bench provides two, both against the real
signal path (cw.envelope statistics, cw_lm rescore):

  1. COPY-FLOOR CURVE (synthetic, ground-truth labels) - sweep noise (and fade)
     over domain-randomized CW (realistic Rician envelope per cw_synth.render),
     decode, and measure CER vs SNR. The single number is the NOISE FLOOR: the
     highest noise sigma at which mean CER stays <= 0.5 (readable). Higher floor
     = copies weaker. A second number, CLEAN CER, guards against regressing easy
     copy (must stay ~0).

  2. REAL RECALL (harvested captures, DB-verified callsigns as ground truth) -
     decode every capture that carries a verified callsign and check the call is
     still recovered. This is the no-regression gate on real signals: an
     experimental front-end that copies weaker synth but drops real callsigns is
     a FALSE win (the neural-net lesson - see morse_ai_overnight memory).

Usage:
  python weak_bench.py                       # bench all decoders, print report
  python weak_bench.py --decoder auto        # one decoder
  python weak_bench.py --json                # machine-readable (for the loop)

The bench NEVER touches the SDR - pure CPU on synth + saved captures - so it runs
safely alongside the harvester (single-tenant SDR law).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cw
import cw_lm
import cw_synth

FS_ENV = 8000.0
HARVEST = HERE.parent / "lab" / "cw_harvest"

# The decoders under test. Each is (env, aud) -> (text, info); we LM-rescore the
# text so the bench measures the full production output, not raw run-lengths.
DECODERS = {
    "classic": cw.decode_env,
    "mf": cw.decode_env_mf,
    "auto": cw.decode_env_auto,
    "mf2": getattr(cw, "decode_env_mf2", cw.decode_env_mf),
    "mf3": getattr(cw, "decode_env_mf3", cw.decode_env_mf),
}


def _decode(fn, env, aud, lm=True):
    try:
        txt, info = fn(env, aud)
    except Exception as e:
        return "", {"err": str(e)}
    if lm:
        try:
            txt = cw_lm.rescore(txt)
        except Exception:
            pass
    return txt, info


# ---- 1. synthetic copy-floor curve ---------------------------------------

# deterministic text pool (real ham phrasing so the LM helps as it would live)
_TEXTS = ["CQ CQ DE W1AW W1AW K", "PARIS PARIS", "DE K4XH UR RST 599 599",
          "TEST DE N7TE N7TE", "73 GL ES TU DE AE6N", "QRZ DE KD0E K",
          "5NN TU GL", "CQ DX DE NW2E NW2E K", "R FB OM UR 579 IN CO",
          "DE NE4U NAME IS SAM SAM"]
_WPMS = [15, 18, 20, 24, 28]


def copy_floor(fn, noises=None, n_per=10, fade=0.0, fade_hz=0.3, lm=True):
    """Mean CER at each noise sigma; noise floor = highest noise with CER<=0.5."""
    if noises is None:
        noises = [0.10, 0.18, 0.28, 0.40, 0.55, 0.75, 1.00, 1.30]
    curve = []
    rng = np.random.default_rng(12345)
    for ns in noises:
        cers = []
        for i in range(n_per):
            text = _TEXTS[i % len(_TEXTS)]
            wpm = _WPMS[i % len(_WPMS)]
            seed = 1000 + int(ns * 1000) * 97 + i        # deterministic, condition-unique
            env, fs = cw_synth.render(text, wpm=wpm, fs=FS_ENV, noise=ns,
                                      fade=fade, fade_hz=fade_hz,
                                      jitter=0.10, seed=seed)
            dec, _ = _decode(fn, env, fs, lm=lm)
            cers.append(cw_synth._cer(dec, text))
        curve.append((ns, float(np.mean(cers))))
    # noise floor: highest noise whose mean CER <= 0.5 (linear-interp the crossing)
    floor = noises[0]
    prev = None
    for ns, c in curve:
        if c <= 0.5:
            floor = ns
        else:
            if prev is not None and prev[1] <= 0.5 and c > 0.5:
                # interpolate crossing between prev and this
                frac = (0.5 - prev[1]) / (c - prev[1] + 1e-9)
                floor = prev[0] + frac * (ns - prev[0])
            break
        prev = (ns, c)
    clean = curve[0][1]                                  # CER at the lowest noise
    return {"curve": curve, "noise_floor": round(float(floor), 3),
            "clean_cer": round(clean, 3)}


# ---- 2. real-capture recall (verified callsigns = ground truth) -----------

def _load_iq(cs16):
    raw = np.fromfile(cs16, dtype=np.int16).astype(np.float32) / 32768.0
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


def _best_eye_offset(iq, fs=250000.0, span_hz=3000, step=50):
    """Aim at the carrier with the OPEN EYE, not the strongest power bin
    (find_offset picks power; a fading strong carrier has power but no eye).
    This is what a good labeler/panel should do; here it makes real_recall a
    fair test of the decoder rather than of find_offset's aim."""
    import cw_quality
    best_e, best_o = -1.0, 0.0
    for off in range(-span_hz, span_hz + 1, step):
        env, _ = cw.envelope(iq, fs, float(off))
        e = cw_quality.eye_opening(env)[0]
        if e > best_e:
            best_e, best_o = e, float(off)
    return best_o, best_e


def real_recall(fn, lm=True, min_eye=3.0, limit=None, aim="eye"):
    """Decode captures carrying verified callsigns; fraction of calls recovered."""
    cases = []
    for j in sorted(glob.glob(str(HARVEST / "*.json"))):
        try:
            d = json.load(open(j))
        except Exception:
            continue
        vc = [v["call"] for v in d.get("verified_calls", [])]
        if vc and float(d.get("eye", 0)) >= min_eye:
            cs16 = HARVEST / d["iq_file"]
            if cs16.exists():
                cases.append((cs16, vc))
    if limit:
        cases = cases[:limit]
    hit = tot = 0
    misses = []
    cache = {}
    cf = HARVEST / "_besteye.json"
    if aim == "eye" and cf.exists():
        try:
            cache = json.load(open(cf))
        except Exception:
            cache = {}
    for cs16, calls in cases:
        iq = _load_iq(str(cs16))
        if aim == "eye":
            off = cache.get(cs16.name)
            if off is None:
                off, _ = _best_eye_offset(iq)
        else:
            off = cw.find_offset(iq, 250000.0, 3000)
        env, aud = cw.envelope(iq, 250000.0, float(off))
        dec, _ = _decode(fn, env, aud, lm=lm)
        up = dec.upper().replace(" ", "")
        for c in calls:
            tot += 1
            if c in dec.upper() or c in up:
                hit += 1
            else:
                misses.append((c, cs16.name))
    return {"recall": round(hit / tot, 3) if tot else None, "hit": hit, "total": tot,
            "n_caps": len(cases), "misses": misses[:12]}


def multi_signal_floor(fn, narrow_bw=None, jitter=0.22, n_per=6):
    """HONEST realistic bench (H12): decode a target CW carrier with a SECOND
    interfering carrier +400-600 Hz at 0.8 amplitude + heavy fist jitter/weight -
    i.e. a crowded band, not a lab tone. Returns the noise sigma at which mean CER
    first exceeds 0.5. narrow_bw sets a CW filter centered on the target (cur_off);
    None = today's wide envelope. This is the metric that stops rewarding single-
    carrier mirages: wide decoders break ~0.3 (they superimpose the QRM), a centered
    narrow filter rejects it and holds to ~1.8 - the CW-filter feature's real value."""
    tgt = ["CQ CQ DE W1AW K", "DE K4XH RST 599", "73 GL DE AE6N K", "QRZ DE KD0E K"]
    qrm = ["DE N7TE N7TE K", "CQ DX UP 5", "599 TU 73 GL"]
    for ns in [0.3, 0.5, 0.8, 1.2, 1.8]:
        cers = []
        for i in range(n_per):
            off2 = 400 + (i * 37) % 200
            iq1, fsq, f0 = cw_synth.render_iq(tgt[i % 4], wpm=[16, 22, 20, 26][i % 4],
                                              fs_iq=12000.0, f0=700.0, noise=0.0,
                                              jitter=jitter, weight=1.0 + ((i % 5 - 2) * 0.06),
                                              seed=300 + int(ns * 100) + i)
            iq2, _, _ = cw_synth.render_iq(qrm[i % 3], wpm=[16, 22, 20, 26][i % 4] + 4,
                                           fs_iq=12000.0, f0=700.0 + off2, noise=0.0,
                                           jitter=jitter, seed=307 + int(ns * 100) + i)
            L = len(iq1)
            iq2 = iq2[:L] if len(iq2) >= L else np.pad(iq2, (0, L - len(iq2)))
            rng = np.random.default_rng(313 + int(ns * 100) + i)
            n = rng.normal(0, ns, L) + 1j * rng.normal(0, ns, L)
            iq = (iq1 + 0.8 * iq2 + n).astype(np.complex64)
            if narrow_bw:
                env, aud = cw.envelope2(iq, fsq, f0, aud=8000, bw_hz=narrow_bw)
            else:
                env, aud = cw.envelope(iq, fsq, f0)
            cers.append(cw_synth._cer(_decode(fn, env, aud)[0], tgt[i % 4]))
        if np.mean(cers) > 0.5:
            return round(ns, 2)
    return 1.8


def bench_one(name, fn, do_real=True, do_fade=True):
    r = {"decoder": name}
    r["snr"] = copy_floor(fn, fade=0.0)
    if do_fade:
        r["fade"] = copy_floor(fn, noises=[0.10, 0.18, 0.28, 0.40, 0.55], fade=0.6)
    if do_real:
        r["real"] = real_recall(fn)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", choices=list(DECODERS), help="just one")
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--no-fade", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    names = [a.decoder] if a.decoder else list(DECODERS)
    out = [bench_one(n, DECODERS[n], do_real=not a.no_real, do_fade=not a.no_fade) for n in names]
    if a.json:
        print(json.dumps(out, indent=2))
        return
    for r in out:
        print("=" * 64)
        print(f"DECODER: {r['decoder']}")
        s = r["snr"]
        print(f"  SNR copy-floor (no fade): noise_floor={s['noise_floor']}  clean_cer={s['clean_cer']}")
        print("    noise:CER  " + "  ".join(f"{ns:.2f}:{c:.2f}" for ns, c in s["curve"]))
        if "fade" in r:
            f = r["fade"]
            print(f"  FADE copy-floor (fade=0.6): noise_floor={f['noise_floor']}  clean_cer={f['clean_cer']}")
            print("    noise:CER  " + "  ".join(f"{ns:.2f}:{c:.2f}" for ns, c in f["curve"]))
        if "real" in r:
            rr = r["real"]
            print(f"  REAL recall: {rr['recall']} ({rr['hit']}/{rr['total']} calls over {rr['n_caps']} caps)")
            if rr["misses"]:
                print(f"    misses: {rr['misses']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
