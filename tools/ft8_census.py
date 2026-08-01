#!/usr/bin/env python3
"""ft8_census.py - FT8 activity census over the EXISTING harvest IQ (no SDR).

Every HF FT8 frequency happens to sit inside our 250 kHz CW-band capture spans,
so the corpus doubles as an FT8 archive. Before building the FT8 adapter (task
list: engine-adapter, FT8 first), measure what's actually there: band-limited
power in the FT8 window (3 kHz at the standard frequency) vs an adjacent quiet
window, plus a dense-multicarrier score (FT8 = many ~50 Hz signals packed
together). Output: per-band / per-hour activity atlas -> lab/ft8_atlas.json.
"""
import glob
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HARVEST = HERE.parent / "lab" / "cw_harvest"
FS = 250_000.0  # rate-ok: offline census of archival 250k harvest captures
FT8 = {7025: 7074, 10120: 10136, 14025: 14074, 3560: 3573, 18075: 18100,
       21025: 21074, 24905: 24915, 28025: 28074}          # capture center -> FT8 kHz


def band_power(iq, off_hz, width_hz):
    """Mean PSD (dB) in [off, off+width] relative to capture center."""
    N = 1 << 15
    m = len(iq) // N * N
    if m < N:
        return None
    seg = iq[:m].reshape(-1, N) * np.hanning(N).astype(np.float32)
    P = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2).mean(0)
    binhz = FS / N
    c = N // 2
    a = c + int(off_hz / binhz)
    b = c + int((off_hz + width_hz) / binhz)
    if not (0 <= a < b < N):
        return None
    return P[a:b]


def census(max_caps=200):
    js = [j for j in sorted(glob.glob(str(HARVEST / "*.json"))) if "besteye" not in j]
    # spread the sample across the whole run
    js = js[:: max(1, len(js) // max_caps)][:max_caps]
    atlas = {}
    t0 = time.time()
    n_done = 0
    for jp in js:
        try:
            d = json.loads(Path(jp).read_text())
            khz = int(d.get("khz", 0))
            if khz not in FT8 or "iq_file" not in d:
                continue
            p = HARVEST / d["iq_file"]
            raw = np.fromfile(str(p), np.int16).astype(np.float32)[: int(2 * 20 * FS)]
            iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
            off = (FT8[khz] - khz) * 1000.0
            sig = band_power(iq, off, 3000.0)                 # the FT8 window
            ref = band_power(iq, off + 4000.0, 3000.0)        # adjacent reference
            if sig is None or ref is None:
                continue
            snr_db = 10 * np.log10(sig.mean() / max(ref.mean(), 1e-12))
            # dense-multicarrier score: fraction of ~60Hz-wide slots above 2x window median
            binhz = FS / (1 << 15)
            slot = max(1, int(60 / binhz))
            med = np.median(sig)
            nslots = len(sig) // slot
            occ = sum(1 for k in range(nslots)
                      if sig[k * slot:(k + 1) * slot].mean() > 2 * med) / max(nslots, 1)
            hh = d["iq_file"].split("T")[1][:2] if "T" in d["iq_file"] else "??"
            key = f"{FT8[khz]}@{hh}Z"
            e = atlas.setdefault(key, {"n": 0, "snr_db": [], "occ": []})
            e["n"] += 1
            e["snr_db"].append(round(float(snr_db), 1))
            e["occ"].append(round(float(occ), 2))
            n_done += 1
        except Exception:
            continue
    # summarize
    out = {}
    for k, e in atlas.items():
        out[k] = {"n": e["n"], "snr_db_med": round(float(np.median(e["snr_db"])), 1),
                  "occ_med": round(float(np.median(e["occ"])), 2)}
    (HERE.parent / "lab" / "ft8_atlas.json").write_text(json.dumps(out, indent=1))
    print(f"[ft8] census: {n_done} windows measured in {(time.time()-t0)/60:.1f} min")
    print(f"{'FT8 freq@hour':16s} {'n':>3s} {'SNR dB':>7s} {'occupancy':>9s}")
    for k in sorted(out, key=lambda k: -out[k]["snr_db_med"])[:18]:
        o = out[k]
        print(f"{k:16s} {o['n']:3d} {o['snr_db_med']:7.1f} {o['occ_med']:9.2f}")
    active = [k for k, o in out.items() if o["snr_db_med"] >= 6 and o["occ_med"] >= 0.3]
    print(f"[ft8] VERDICT: {len(active)}/{len(out)} band-hours show strong FT8 activity"
          f" -> the adapter has {'plenty' if active else 'little'} to eat. "
          f"-> lab/ft8_atlas.json")


if __name__ == "__main__":
    census()
