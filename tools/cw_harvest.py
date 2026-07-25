#!/usr/bin/env python3
"""cw_harvest.py - the CW trip-wire recorder.

Sits on a CW sub-band and records NOTHING until real Morse appears; when a carrier's
eye opens (copyable CW), it trips, records the full-band IQ until the signal is over,
labels it (our decode + LM + DB-verified callsigns), and goes back to waiting. So it
only ever spends disk on real signals - perfect for unattended overnight harvesting
of a REAL corpus (validation + real-noise harvest for training the neural decoder).

  python cw_harvest.py --khz 7025 --hours 8      # watch 40m CW overnight
  python cw_harvest.py --khz 14030 --antenna "Antenna C"

Laws honored (from the rig's memory): deadline-guarded reads (no hang on a wedged
SDR), radio_lock, and STOP THE WARDEN first (it fights for the dial). Launch
detached (Start-Process) to survive the session; it won't survive a reboot.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cw
import cw_lm
import cw_quality
import hamdb

FS = 250_000.0
OUT = HERE.parent / "lab" / "cw_harvest"
# BIG-CORPUS MODE (2026-07-24, user: "we have a terabyte, save everything, feed it
# to Fable 5 as training data"). Capture LONGER (whole QSOs -> context + repeated
# callsigns for the soft-decision / neural work) and grab more of the quality range.
EYE_TRIP = 1.8          # a touch lower -> catch more marginal CW for training diversity
#                         (still above pure noise so we don't fill disk with static)
HANG_S = 20.0           # keep recording through longer gaps so a QSO isn't split
MAX_REC_S = 600.0       # up to 10 min per capture (250 kHz cs16 ~1 MB/s -> ~600 MB max)
CHUNK_S = 4.0           # monitor/record granularity
MIN_FREE_FRAC = 0.10    # FILL THE DISK TO 90% (user: capture all the Morse we can;
#                         we only ever record when the eye is open, so it's all real
#                         CW, not static). Stop taking NEW captures at 10% free so the
#                         OS/drive don't choke; existing captures always finish.


def _gmt():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _disk():
    """(free_GB, free_fraction). If we can't tell, report 'plenty' so we never block."""
    import shutil
    try:
        u = shutil.disk_usage(str(OUT.anchor or OUT))
        return u.free / 1e9, u.free / u.total
    except Exception:
        return 1e9, 1.0


def best_eye(iq, khz_center):
    """Scan the CW sub-band in this chunk; return (best_eye, best_off_hz)."""
    # reuse the panel's detector idea: peaks in the spectrum, eye per carrier
    N = 1 << 13
    m = len(iq) // N * N
    if m < N:
        return 0.0, 0.0
    seg = iq[:m].reshape(-1, N) * np.hanning(N).astype(np.float32)
    P = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2).mean(0)
    binhz = FS / N
    c = N // 2
    span = int(3000 / binhz)                       # +/-3 kHz around center
    noise = np.median(P)
    band = P[c - span:c + span]
    peaks = [(i - span) * binhz for i in range(2, len(band) - 2)
             if band[i] > noise * 6 and band[i] >= band[i - 1] and band[i] > band[i + 1]]
    best_e, best_o = 0.0, 0.0
    for off in peaks[:8]:
        env, aud = cw.envelope(iq, FS, off)
        e = cw_quality.eye_opening(env)[0]
        if e > best_e:
            best_e, best_o = e, off
    return best_e, best_o


def label_and_save(iq, khz, meta):
    """Decode + DB-verify the recording, write .cs16 + .json sidecar."""
    OUT.mkdir(parents=True, exist_ok=True)
    off = cw.find_offset(iq, FS, 3000)
    env, aud = cw.envelope(iq, FS, off)
    raw, info = cw.decode_env_auto(env, aud)
    text = cw_lm.rescore(raw)
    eye = cw_quality.eye_opening(env)[0]
    # DB-verify callsigns ONLY when the eye is actually open (>= readable ~3.0).
    # Below that the decode is noise and any callsign match is a coincidence -
    # false-verifying garbage would pollute the high-score system.
    verified = []
    if eye >= 3.0:
        for tok in set(text.split()):
            if cw_lm.CALL_RE.match(tok) and any(ch.isdigit() for ch in tok):
                r = hamdb.verify(tok)
                if r.get("status") == "VALID":
                    verified.append({"call": tok, "name": r.get("name", ""), "qth": r.get("qth", "")})
    base = OUT / f"cw_{int(khz)}_{_gmt()}"
    # cs16 interleaved int16
    inter = np.empty(2 * len(iq), np.int16)
    inter[0::2] = np.clip(iq.real * 32768, -32768, 32767).astype(np.int16)
    inter[1::2] = np.clip(iq.imag * 32768, -32768, 32767).astype(np.int16)
    inter.tofile(str(base) + ".cs16")
    # quality tier for training stratification: HIGH (clean, labelable) / MID / LOW (rough)
    quality = "high" if eye >= 3.5 else ("mid" if eye >= 2.5 else "low")
    label_conf = "verified" if verified else ("decode" if eye >= 3.0 else "none")
    sidecar = {"iq_file": base.name + ".cs16", "khz": khz, "secs": round(len(iq) / FS, 1),
               "text": text, "raw": raw, "wpm": info.get("wpm", 0), "eye": round(float(eye), 2),
               "quality": quality, "label_conf": label_conf,
               "verified_calls": verified, **meta}
    (Path(str(base) + ".json")).write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    tag = f" calls={[v['call'] for v in verified]}" if verified else ""
    print(f"[harvest] SAVED {base.name}.cs16  {len(iq)/FS:.0f}s eye={eye:.1f} "
          f"wpm={info.get('wpm',0)} '{text[:40]}'{tag}", flush=True)
    return sidecar


# CW sub-band calling areas across HF (kHz) - the channels to scour for Morse
SCAN_BANDS = [7025, 10120, 14025, 3560, 18075, 21025, 24905, 28025, 5357]
DWELL_CHUNKS = 3        # monitor chunks per band before hopping (when idle)


def run(khz, hours, antenna, scan=True):
    try:
        import radio_lock
        lock = radio_lock.acquire("cw_harvest", "harvest", priority=40, wait_s=20)
    except Exception:
        lock = True
    if not lock:
        print("[harvest] SDR busy (panel/warden holds it) - stop them first."); return
    from SoapySDR import SOAPY_SDR_RX
    sdr, st = cw._open_sdr(antenna, FS)
    bands = SCAN_BANDS if scan else [khz]
    bi = 0
    sdr.setFrequency(SOAPY_SDR_RX, 0, bands[0] * 1e3)
    time.sleep(0.2)
    print(f"[harvest] {'SCANNING ' + str(len(bands)) + ' CW bands' if scan else 'watching ' + str(khz)} "
          f"on {antenna}, trip eye>={EYE_TRIP}, for {hours}h. Records only when CW appears.", flush=True)
    deadline = time.time() + hours * 3600
    recording, buf, last_open, n_saved, dwell = False, [], 0.0, 0, 0
    cur = bands[0]
    try:
        while time.time() < deadline:
            iq = cw._grab(sdr, st, CHUNK_S, FS)
            if len(iq) < FS:
                continue
            eye, off = best_eye(iq, cur)
            now = time.time()
            if not recording:
                free_gb, free_frac = _disk()
                if eye >= EYE_TRIP and free_frac >= MIN_FREE_FRAC:   # TRIP - CW found
                    recording, buf, last_open = True, [iq], now
                    print(f"[harvest] TRIP eye={eye:.1f} @ {cur} kHz - recording... "
                          f"(disk {free_gb:.0f} GB / {free_frac*100:.0f}% free)", flush=True)
                elif eye >= EYE_TRIP:                     # would trip but disk ~90% full
                    print(f"[harvest] DISK 90% FULL ({free_gb:.0f} GB free) - skipping capture", flush=True)
                elif scan:                                # keep scanning the bands
                    dwell += 1
                    if dwell >= DWELL_CHUNKS:
                        dwell = 0; bi = (bi + 1) % len(bands); cur = bands[bi]
                        sdr.setFrequency(SOAPY_SDR_RX, 0, cur * 1e3); time.sleep(0.15)
            else:
                buf.append(iq)
                if eye >= EYE_TRIP:
                    last_open = now
                dur = sum(len(b) for b in buf) / FS
                if now - last_open > HANG_S or dur > MAX_REC_S:   # signal over / capped
                    label_and_save(np.concatenate(buf), cur,
                                   {"trip_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                    recording, buf, n_saved = False, [], n_saved + 1
    finally:
        try:
            sdr.deactivateStream(st); sdr.closeStream(st)
        except Exception:
            pass
        try:
            radio_lock.release("cw_harvest")
        except Exception:
            pass
        print(f"[harvest] done. {n_saved} recording(s) saved to {OUT}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--khz", type=float, default=7025)      # used when --no-scan
    ap.add_argument("--hours", type=float, default=10)
    ap.add_argument("--antenna", default="Antenna C")
    ap.add_argument("--no-scan", action="store_true")
    a = ap.parse_args()
    run(a.khz, a.hours, a.antenna, scan=not a.no_scan)


if __name__ == "__main__":
    main()
