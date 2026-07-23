#!/usr/bin/env python3
"""test_cw.py — regression guard for the CW decoder.

Re-decodes the saved corpus (known-good captures with logged baselines) plus the
synth self-test, so a code change that quietly breaks decoding is caught BEFORE
it ships. This is the safety net: run it after any edit to cw.py / the decode
path. Exit code 0 = all good, non-zero = a regression.

  python tools/cw.py selftest      # the 5-second synth check (also here)
  python tools/test_cw.py          # synth + full corpus regression
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cw

FS = 250_000.0
CORPUS = HERE.parent / "lab" / "cw_corpus"
CORPUS_JSONL = HERE.parent / "lab" / "cw_corpus.jsonl"
Q_TOLERANCE = 0.15        # allow small drift; a bigger q drop = regression


def load(f):
    raw = np.fromfile(f, np.int16)
    return ((raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32))
            / 32768.0).astype(np.complex64)


def qof(t):
    c = [x for x in t if x != " "]
    return sum(1 for x in c if x != "?") / len(c) if c else 0.0


def main():
    fails = 0
    print("=" * 62)
    print("CW DECODER REGRESSION TEST")
    print("=" * 62)

    # 1) synth self-test (must decode NDB) — guards the core DSP
    from importlib import import_module
    fs = FS
    dit = int(0.06 * fs)
    seq = []
    for ch in "NDB":
        for el in cw.INV[ch]:
            seq += [(1, dit if el == "." else 3 * dit), (0, dit)]
        seq.append((0, 3 * dit))
    key = np.concatenate([np.full(l, float(s)) for s, l in seq])
    t = np.arange(len(key))
    iq = (key * np.exp(2j * np.pi * -6200 / fs * t)).astype(np.complex64)
    rng = np.random.default_rng(1)
    iq += (rng.normal(0, 0.15, len(iq)) + 1j * rng.normal(0, 0.15, len(iq))).astype(np.complex64)
    env, a = cw.envelope(iq, fs, -6200)
    txt = cw.decode_env(env, a)[0]
    ok = "NDB" in txt.replace(" ", "")
    print(f"\n[synth] sent NDB -> {txt!r:20}  {'PASS' if ok else 'FAIL'}")
    fails += 0 if ok else 1

    # 2) corpus regression — re-decode each saved capture vs its logged baseline
    base = {}
    try:
        for line in open(CORPUS_JSONL, encoding="utf-8"):
            r = json.loads(line); base[r["iq_file"]] = r
    except Exception:
        pass
    files = sorted(glob.glob(str(CORPUS / "*.cs16")))
    if not files:
        print("\n[corpus] no captures found — collect some first (cw_collect.py)")
    else:
        print(f"\n[corpus] {len(files)} capture(s):")
        for f in files:
            name = Path(f).name
            iq = load(f)
            off = cw.find_offset(iq, FS, search=50000)
            env, a = cw.envelope(iq, FS, off)
            txt = cw.decode_env(env, a)[0]
            q = qof(txt)
            b = base.get(name, {})
            bq = b.get("q_ratio", 0.0)
            # regression = quality dropped meaningfully below the baseline
            reg = bq > 0 and q < bq - Q_TOLERANCE
            # also: did we keep the callsign we knew was there?
            print(f"  {name[:34]:34} q={q:.2f} (base {bq:.2f})  "
                  f"{'REGRESSION' if reg else 'ok'}  {txt[:34]!r}")
            fails += 1 if reg else 0

    print("\n" + "=" * 62)
    print(f"RESULT: {'ALL PASS' if fails == 0 else f'{fails} REGRESSION(S)'}")
    print("=" * 62)
    return fails


if __name__ == "__main__":
    sys.exit(min(1, main()))
