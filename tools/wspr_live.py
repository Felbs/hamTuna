#!/usr/bin/env python3
"""wspr_live.py - decode WSPR from an IQ capture via the wsprd engine-adapter.

Same engine-adapter pattern as ft8_live.py (task #18): hamTuna doesn't reimplement
WSPR's Fano/soft decoder - it wraps wsprd (WSJT-X). WSPR is a 2-minute mode (110.6 s
transmissions starting on EVEN UTC minutes, ~1400-1600 Hz audio), so this carves a
120 s slot aligned to an even minute from a capture, mixes the band's WSPR dial to
baseband, writes the 12 kHz wav wsprd wants, runs wsprd, and parses the spots
(callsign, grid, power, SNR, freq, drift).

  python wspr_live.py decode <cs16>          # every even-minute slot in a capture
  python wspr_live.py slot <cs16> <t_s>      # one 120 s slot at offset t

NOTE: not wired into the live panel yet - the panel ring is 24 s and WSPR needs
120 s. Panel WSPR wants a dedicated 2-minute capture buffer (a future enhancement);
until then this decodes captures (and is the regression-gated adapter).
"""
import json
import re
import subprocess
import sys
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

HERE = Path(__file__).resolve().parent
FS = 250_000.0  # rate-ok: offline decoder of archival 250k cs16 captures - no
#                 SDR open here; iq_to_wav takes fs= for other-rate callers
AUD = 12_000
WSPRD = r"C:\wsjtx\bin\wsprd.exe"
SLOT_S = 120                        # WSPR period
# WSPR dial frequencies (kHz) per band - the USB dial so WSPR lands at ~1500 Hz.
WSPR_DIAL_KHZ = {"160m": 1836.6, "80m": 3568.6, "60m": 5287.2, "40m": 7038.6,
                 "30m": 10138.7, "20m": 14095.6, "17m": 18104.6, "15m": 21094.6,
                 "12m": 24924.6, "10m": 28124.6, "6m": 50293.0}
LOG = HERE.parent / "lab" / "wspr_log.jsonl"

# wsprd spot line, e.g.:  " 2358 -21  0.4  14.097091  0  DK6UG JN49 37"
_LINE = re.compile(
    r"^\s*(\d{4})\s+([+-]?\d+)\s+([+-]?[\d.]+)\s+([\d.]+)\s+([+-]?\d+)\s+(.+?)\s*$")
_CALL = re.compile(r"\b([A-Z0-9]{1,3}\d[A-Z]{1,4}|<[A-Z0-9/]+>)\b")
_GRID = re.compile(r"\b([A-R]{2}\d{2})\b")


def iq_to_wav(iq, dial_khz, wav_path, off_hz=None, fs=FS, center_khz=None):
    """Mix the WSPR dial to baseband -> 12 kHz mono wav (WSPR at ~1500 Hz). Pass
    off_hz directly, or dial_khz + center_khz to compute it."""
    if off_hz is None:
        off_hz = (dial_khz - center_khz) * 1000.0
    n = np.arange(len(iq), dtype=np.float64)
    x = iq * np.exp(-2j * np.pi * off_hz / fs * n)
    taps = firwin(401, 2800.0 / (fs / 2)).astype(np.float32)
    audio = resample_poly(lfilter(taps, 1.0, x), AUD, int(fs)).real.astype(np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.7
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(AUD)
        w.writeframes((audio[:SLOT_S * AUD] * 32767).astype(np.int16).tobytes())


def decode_wav(wav_path, dial_mhz):
    """Run wsprd, return (records, err). wsprd writes its hashtable to cwd, so we
    run IN the wav's directory and pass the BASENAME (passing the full path with
    cwd set makes wsprd look for parent/parent/file -> 'Cannot open data file')."""
    p = Path(wav_path)
    try:
        # pipe-ok: wsprd emits ~a dozen spot lines parsed in-memory; a 120 s
        # expiry abandons one slot, never a night's product
        out = subprocess.run([WSPRD, "-f", f"{dial_mhz:.6f}", p.name],
                             capture_output=True, text=True, timeout=120,
                             cwd=str(p.parent)).stdout
    except Exception as e:
        return [], f"wsprd error: {e}"
    recs = []
    for ln in out.splitlines():
        m = _LINE.match(ln)
        if not m:
            continue
        msg = m.group(6)
        calls = _CALL.findall(msg)
        grid = _GRID.search(msg)
        recs.append({"snr": int(m.group(2)), "dt": float(m.group(3)),
                     "freq_mhz": float(m.group(4)), "drift": int(m.group(5)),
                     "msg": msg, "calls": calls,
                     "grid": grid.group(1) if grid else None})
    return recs, None


def _load(cs16):
    p = Path(cs16)
    khz = int(p.name.split("_")[1])
    raw = np.fromfile(p, np.int16).astype(np.float32) / 32768.0
    iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    t0 = datetime.strptime(p.name.split("_")[2].split(".")[0],
                           "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return iq, khz, t0


def _band_for(khz):
    """Nearest band whose WSPR dial falls inside a 250 kHz window at center khz."""
    for band, dial in WSPR_DIAL_KHZ.items():
        if abs(dial - khz) < 120:
            return band, dial
    return None, None


def slot(cs16, t0_s):
    iq, khz, _ = _load(cs16)
    band, dial = _band_for(khz)
    if dial is None:
        return [], f"no WSPR dial within the 250 kHz window at {khz} kHz"
    a = int(t0_s * FS)
    seg = iq[a:a + SLOT_S * int(FS)]
    if len(seg) < (SLOT_S - 5) * FS:
        return [], "capture too short for a 120 s WSPR slot at that offset"
    off = (dial - khz) * 1000.0
    wav = HERE.parent / "lab" / "_wspr_slot.wav"
    wav.parent.mkdir(exist_ok=True)
    iq_to_wav(seg, dial, wav, off_hz=off)
    return decode_wav(wav, dial / 1000.0)


def decode(cs16):
    iq, khz, t0 = _load(cs16)
    dur = len(iq) / FS
    # first even-minute boundary in the capture
    sec_into = (t0.minute % 2) * 60 + t0.second + t0.microsecond / 1e6
    t = (SLOT_S - sec_into) % SLOT_S
    LOG.parent.mkdir(exist_ok=True)
    total = 0
    while t + SLOT_S <= dur:
        recs, err = slot(str(cs16), t)
        stamp = (t0 + timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if err:
            print(f"[wspr] {stamp}: {err}", flush=True); break
        for r in recs:
            r["utc"] = stamp
            with open(LOG, "a") as f:
                f.write(json.dumps(r) + "\n")
        if recs:
            print(f"[wspr] {stamp}: {len(recs)} spots  e.g. {recs[0]['msg']!r}", flush=True)
        total += len(recs)
        t += SLOT_S
    print(f"[wspr] {total} WSPR spots -> {LOG.name}", flush=True)
    return total


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "slot":
        recs, err = slot(sys.argv[2], float(sys.argv[3]))
        print(err or f"{len(recs)} spots:")
        for r in recs:
            print(f"  {r['snr']:+3d} dB  {r['freq_mhz']:.6f}  {r['msg']}")
    elif len(sys.argv) >= 3 and sys.argv[1] == "decode":
        decode(sys.argv[2])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
