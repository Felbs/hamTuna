#!/usr/bin/env python3
"""ft8_live.py - decode FT8 from a live IQ snapshot and log the contacts.

The engine-adapter (task #18): hamTuna doesn't reimplement FT8 - it wraps the
world-class jt9 decoder (WSJT-X). This mixes the FT8 sub-band of a 250 kHz
capture down to a 12 kHz USB audio wav, runs jt9, parses the decodes into
structured records (utc, snr, dt, freq, message, call, grid), and appends them
to lab/ft8_log.jsonl. Same pattern the panel will call once per 15 s slot to
show live FT8 + feed the logbook.

  python ft8_live.py decode <cs16>       # decode every 15 s slot in a capture
  python ft8_live.py slot <cs16> <t_s>   # one 15 s slot at offset t
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
FS = 250_000.0
AUD = 12_000
JT9 = r"C:\wsjtx\bin\jt9.exe"
DIAL_OFF = {14025: 49_000.0, 7025: 49_000.0, 3560: 13_000.0, 18075: 25_000.0,
            21025: 49_000.0, 10120: 16_000.0, 24905: 10_000.0, 28025: 49_000.0}
LOG = HERE.parent / "lab" / "ft8_log.jsonl"
# jt9 line: "HHMMSS  snr  dt  freq ~  message"
_LINE = re.compile(r"^(\d{6})\s+([+-]?\d+)\s+([+-]?[\d.]+)\s+(\d+)\s+~?\s+(.*\S)")
_CALL = re.compile(r"\b([A-Z0-9]{1,3}\d[A-Z]{1,4})\b")
_GRID = re.compile(r"\b([A-R]{2}\d{2})\b")


def iq_to_wav(iq, khz, wav_path):
    """Mix the FT8 dial to baseband, USB-filter to 12 kHz mono wav."""
    off = DIAL_OFF.get(khz, 49_000.0)
    n = np.arange(len(iq), dtype=np.float64)
    x = iq * np.exp(-2j * np.pi * off / FS * n)
    taps = firwin(401, 3300.0 / (FS / 2)).astype(np.float32)
    audio = resample_poly(lfilter(taps, 1.0, x), AUD, int(FS)).real.astype(np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * 0.7
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(AUD)
        w.writeframes((audio[:15 * AUD] * 32767).astype(np.int16).tobytes())


def decode_wav(wav_path):
    """Run jt9, return parsed FT8 records."""
    try:
        out = subprocess.run([JT9, "-8", "-d", "3", str(wav_path)],
                             capture_output=True, text=True, timeout=45).stdout
    except Exception as e:
        return [], f"jt9 error: {e}"
    recs = []
    for ln in out.splitlines():
        m = _LINE.match(ln.strip())
        if not m:
            continue
        msg = m.group(5)
        calls = _CALL.findall(msg)
        grid = _GRID.search(msg)
        recs.append({"snr": int(m.group(2)), "dt": float(m.group(3)),
                     "audio_hz": int(m.group(4)), "msg": msg,
                     "calls": calls, "grid": grid.group(1) if grid else None})
    return recs, None


def slot(cs16, t0_s):
    p = Path(cs16)
    khz = int(p.name.split("_")[1])
    raw = np.fromfile(p, np.int16).astype(np.float32) / 32768.0
    iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    a = int(t0_s * FS)
    seg = iq[a:a + 15 * int(FS)]
    if len(seg) < 15 * FS:
        return [], "capture too short for a 15 s slot at that offset"
    wav = HERE.parent / "lab" / "_ft8_slot.wav"
    iq_to_wav(seg, khz, wav)
    return decode_wav(wav)


def decode(cs16):
    """Every slot-aligned 15 s window in the capture."""
    p = Path(cs16)
    khz = int(p.name.split("_")[1])
    t0 = datetime.strptime(p.name.split("_")[2].split(".")[0], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    dur = p.stat().st_size / 4 / FS
    sec = (t0.minute * 60 + t0.second) % 15
    t = (15 - sec) % 15
    LOG.parent.mkdir(exist_ok=True)
    total = calls = 0
    while t + 15 <= dur:
        recs, err = slot(str(p), t)
        stamp = (t0 + timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for r in recs:
            r["utc"] = stamp
            with open(LOG, "a") as f:
                f.write(json.dumps(r) + "\n")
            calls += len(r["calls"])
        if recs:
            print(f"[ft8] {stamp}: {len(recs)} decodes"
                  f"  e.g. {recs[0]['msg']!r}", flush=True)
        total += len(recs)
        t += 15
    print(f"[ft8] {total} FT8 decodes ({calls} callsigns) -> {LOG.name}", flush=True)
    return total


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "slot":
        recs, err = slot(sys.argv[2], float(sys.argv[3]))
        print(err or f"{len(recs)} decodes:")
        for r in recs:
            print(f"  {r['snr']:+3d} dB  {r['msg']}")
    elif len(sys.argv) >= 3 and sys.argv[1] == "decode":
        decode(sys.argv[2])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
