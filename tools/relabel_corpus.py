#!/usr/bin/env python3
"""relabel_corpus.py - batch re-label the harvest corpus for neural training.

The harvester labeled each capture at find_offset (power-argmax) - the 18:05 audit
showed a best-EYE aim + mf2 beats that ~15% of the time and mis-aims garble labels.
This pass re-decodes every capture at its best-eye carrier and writes the result
into the sidecar under "relabel" (originals untouched), tagging the training tier:

  relabel: {off_hz, eye, text, quality(high/mid/low), label_conf, verified_calls}

Safe alongside the running harvester: skips files modified in the last 2 minutes
(never races a capture being written), only touches *.json sidecars it has read.
Run:  python tools/relabel_corpus.py            (resumable - skips already-done)
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cw
import cw_lm
import cw_quality

FS = 250_000.0  # rate-ok: offline relabel of the archival 250k corpus, no SDR
HARVEST = HERE.parent / "lab" / "cw_harvest"


def load_iq(p):
    r = np.fromfile(p, np.int16).astype(np.float32) / 32768.0
    return (r[0::2] + 1j * r[1::2]).astype(np.complex64)


def best_eye_off(iq):
    """Top power peaks -> judge each by eye -> openest wins (the honest aim)."""
    N = 1 << 13
    m = len(iq) // N * N
    if m < N:
        return 0.0, 0.0
    seg = iq[:m].reshape(-1, N) * np.hanning(N).astype(np.float32)
    P = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2).mean(0)
    bh = FS / N
    c = N // 2
    span = int(3000 / bh)
    band = P[c - span:c + span]
    idx = [i for i in range(1, len(band) - 1)
           if band[i] >= band[i - 1] and band[i] > band[i + 1]]
    idx.sort(key=lambda i: -band[i])
    be, bo = -1.0, 0.0
    for i in idx[:8]:
        off = (i - span) * bh
        env, _ = cw.envelope(iq, FS, off)
        e = cw_quality.eye_opening(env)[0]
        if e > be:
            be, bo = e, off
    return bo, max(be, 0.0)


def main():
    js = [j for j in sorted(glob.glob(str(HARVEST / "*.json"))) if "besteye" not in j]
    done = skip = err = 0
    t0 = time.time()
    for jp in js:
        try:
            if time.time() - os.path.getmtime(jp) < 120:
                skip += 1; continue                      # harvester may still be writing
            d = json.load(open(jp, encoding="utf-8"))
            if "iq_file" not in d or "relabel" in d:
                skip += 1; continue                      # resumable
            cs = HARVEST / d["iq_file"]
            if not cs.exists():
                skip += 1; continue
            iq = load_iq(str(cs))
            off, eye = best_eye_off(iq)
            env, aud = cw.envelope(iq, FS, off)
            raw, info = cw.decode_env_auto(env, aud)
            text = cw_lm.rescore(raw)
            quality = "high" if eye >= 3.5 else ("mid" if eye >= 2.5 else "low")
            verified = []
            if eye >= 3.0:                               # same open-eye law as the harvester
                try:
                    import hamdb
                    for tok in set(text.split()):
                        if cw_lm.CALL_RE.match(tok) and any(ch.isdigit() for ch in tok):
                            r = hamdb.verify(tok)
                            if r.get("status") == "VALID":
                                verified.append({"call": tok, "name": r.get("name", "")})
                except Exception:
                    pass
            d["relabel"] = {"off_hz": round(float(off), 1), "eye": round(float(eye), 2),
                            "text": text, "quality": quality,
                            "label_conf": "verified" if verified else
                                          ("decode" if eye >= 3.0 else "none"),
                            "verified_calls": verified, "route": info.get("route", "")}
            Path(jp).write_text(json.dumps(d, indent=2), encoding="utf-8")
            done += 1
            if done % 50 == 0:
                print(f"[relabel] {done} done, {skip} skipped, {err} err "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
        except Exception as e:
            err += 1
            if err < 6:
                print(f"[relabel] ERR {Path(jp).name}: {e}", flush=True)
    print(f"[relabel] COMPLETE: {done} relabeled, {skip} skipped, {err} errors "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
