#!/usr/bin/env python3
"""ft8_slice.py - carve slot-aligned FT8 wavs out of harvested 250 kHz IQ.

Our CW-band captures at 14025 kHz span +/-125 kHz, so the 20 m FT8 window
(14074.0 dial, audio 0-3.2 kHz USB) rides along at +49 kHz. This mixes it to
baseband, USB-demods to a 12 kHz mono wav, and trims to the 15 s slot grid
(:00/:15/:30/:45 UTC) that jt9 expects. First step of the FT8 engine adapter.

  python ft8_slice.py lab/cw_harvest/cw_14025_<utc>.cs16 out_dir
"""
import sys
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

FS = 250_000.0
DIAL_OFF = {14025: 49_000.0, 7025: 49_000.0, 3560: 13_000.0, 18075: 25_000.0,
            21025: 49_000.0, 10120: 16_000.0, 24905: 10_000.0, 28025: 49_000.0}
AUD = 12_000


def main(cs16, outdir):
    p = Path(cs16)
    khz = int(p.name.split("_")[1])
    off = DIAL_OFF[khz]
    t0 = datetime.strptime(p.name.split("_")[2].split(".")[0],
                           "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    raw = np.fromfile(p, np.int16).astype(np.float32) / 32768.0
    iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    dur = len(iq) / FS
    # mix FT8 dial to DC, low-pass the 0-3.2 kHz USB audio band, real part
    n = np.arange(len(iq), dtype=np.float64)
    x = iq * np.exp(-2j * np.pi * off / FS * n)
    taps = firwin(401, 3300.0 / (FS / 2)).astype(np.float32)
    xf = lfilter(taps, 1.0, x)
    audio = resample_poly(xf, AUD, int(FS)).real.astype(np.float32)
    # slot grid: first :00/:15/:30/:45 boundary after t0
    sec_into = (t0.minute * 60 + t0.second) % 15
    first = (15 - sec_into) % 15
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    made = []
    t = first
    while t + 15.0 <= dur:
        a = int(t * AUD)
        seg = audio[a:a + 15 * AUD]
        seg = seg / (np.max(np.abs(seg)) + 1e-9) * 0.7
        ts = (t0 + timedelta(seconds=t)).strftime("%y%m%d_%H%M%S")
        fn = outdir / f"{ts}.wav"
        with wave.open(str(fn), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(AUD)
            w.writeframes((seg * 32767).astype(np.int16).tobytes())
        made.append(fn)
        t += 15.0
    print(f"{p.name}: {len(made)} slot wavs -> {outdir}")
    return made


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "../lab/ft8_wavs")
