#!/usr/bin/env python3
"""exp13_coherent.py - A/B the coherent CW front-end vs incoherent (EXP-13).

Question: does coherent integration (|sum(x)| before the envelope) copy weaker CW
than the classic incoherent envelope (sum|x|)? Fair test: the SAME synthetic IQ
and the SAME real captures run through both front-ends -> mf2 decode + LM rescore
-> CER / callsign recall. Honest gate: coherent must LIFT the synthetic noise
floor AND not regress real callsign recall (the neural-net lesson: a front-end
that copies weaker synth but drops real calls is a false win). Else it's a kill.

  python exp13_coherent.py
"""
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

HARVEST = HERE.parent / "lab" / "cw_harvest"
_TEXTS = ["CQ CQ DE W1AW W1AW K", "PARIS PARIS", "DE K4XH UR RST 599 599",
          "TEST DE N7TE N7TE", "73 GL ES TU DE AE6N", "QRZ DE KD0E K"]
_WPMS = [15, 18, 20, 24, 28]

FRONTENDS = {
    "incoherent": lambda iq, fs, off: cw.envelope(iq, fs, off),
    "coherent":   lambda iq, fs, off: cw.envelope_coherent(iq, fs, off, coh_ms=12.0),
}


def _decode(env, aud):
    txt, _ = cw.decode_env_mf2(env, aud)
    try:
        txt = cw_lm.rescore(txt)
    except Exception:
        pass
    return txt


def iq_copy_floor(fe, noises=(0.3, 0.5, 0.8, 1.2, 1.8, 2.5), n_per=8, fade=0.0):
    """Mean CER at each noise sigma on synthetic IQ; floor = highest noise with
    mean CER <= 0.5 (linear-interpolated crossing)."""
    curve = []
    for ns in noises:
        cers = []
        for i in range(n_per):
            text, wpm = _TEXTS[i % len(_TEXTS)], _WPMS[i % len(_WPMS)]
            seed = 2000 + int(ns * 1000) * 91 + i
            iq, fs, f0 = cw_synth.render_iq(text, wpm=wpm, fs_iq=12000.0, f0=700.0,
                                            noise=ns, fade=fade, jitter=0.10, seed=seed)
            env, aud = fe(iq, fs, f0)
            cers.append(cw_synth._cer(_decode(env, aud), text))
        curve.append((ns, float(np.mean(cers))))
    floor, prev = curve[0][0], None
    for ns, c in curve:
        if c <= 0.5:
            floor = ns
        else:
            if prev and prev[1] <= 0.5:
                frac = (0.5 - prev[1]) / (c - prev[1] + 1e-9)
                floor = prev[0] + frac * (ns - prev[0])
            break
        prev = (ns, c)
    return {"floor": round(float(floor), 3), "clean": round(curve[0][1], 3), "curve": curve}


def _load_iq(cs16):
    raw = np.fromfile(cs16, np.int16).astype(np.float32) / 32768.0
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


def real_recall(fe, min_eye=3.0):
    """Verified-callsign recall on real captures, aimed with the best-eye cache."""
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
    cache = {}
    cf = HARVEST / "_besteye.json"
    if cf.exists():
        try:
            cache = json.load(open(cf))
        except Exception:
            cache = {}
    hit = tot = 0
    for cs16, calls in cases:
        iq = _load_iq(str(cs16))
        off = cache.get(cs16.name)
        if off is None:
            off = cw.find_offset(iq, 250000.0, 3000)
        env, aud = fe(iq, 250000.0, float(off))
        txt = _decode(env, aud)
        up = txt.upper().replace(" ", "")
        for c in calls:
            tot += 1
            if c in txt.upper() or c in up:
                hit += 1
    return {"recall": round(hit / tot, 3) if tot else None, "hit": hit,
            "total": tot, "n": len(cases)}


def main():
    print("=" * 68)
    print("EXP-13: coherent vs incoherent CW front-end (mf2 + LM, same inputs)")
    print("=" * 68)
    res = {}
    for name, fe in FRONTENDS.items():
        snr = iq_copy_floor(fe)
        fade = iq_copy_floor(fe, noises=(0.3, 0.5, 0.8, 1.2), fade=0.6)
        real = real_recall(fe)
        res[name] = (snr, fade, real)
        print(f"\n[{name:10s}] IQ noise-floor {snr['floor']:.3f} (clean CER {snr['clean']}) | "
              f"fade-floor {fade['floor']:.3f} | real recall {real['recall']} "
              f"({real['hit']}/{real['total']} over {real['n']} caps)")
        print("   snr curve: " + "  ".join(f"{n:.1f}:{c:.2f}" for n, c in snr["curve"]))
    i, c = res["incoherent"], res["coherent"]
    d_floor = c[0]["floor"] - i[0]["floor"]
    d_fade = c[1]["floor"] - i[1]["floor"]
    have_real = (i[2]["total"] > 0)
    d_real = (c[2]["recall"] or 0) - (i[2]["recall"] or 0) if have_real else 0.0
    print("\n" + "=" * 68)
    print(f"DELTA coherent-incoherent:  SNR-floor {d_floor:+.3f}  "
          f"fade-floor {d_fade:+.3f}  real-recall {d_real:+.3f}"
          + ("" if have_real else "  (NO real captures on disk - synthetic-only)"))
    no_regress = (not have_real) or d_real >= -0.01
    clean_ok = c[0]["clean"] <= i[0]["clean"] + 0.05
    win = d_floor > 0.1 and no_regress and clean_ok
    print("VERDICT: " + ("COHERENT WINS - copies weaker, no real regression, keep it"
                         if win else "NO WIN - honest kill (no floor lift / regresses)"))
    print("=" * 68)
    return 0 if win else 2


if __name__ == "__main__":
    sys.exit(main())
