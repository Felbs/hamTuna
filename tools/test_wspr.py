#!/usr/bin/env python3
"""test_wspr.py - regression gate for the WSPR engine-adapter plumbing.

WSPR is sparse (a beacon transmits ~20% of slots, far fewer stations than FT8), so
a real decode can't be a portable hard gate. This gates the parts WE wrote: the
offset-correct front end, that wsprd actually OPENS and RUNS on our wav (the bug
this catches: passing a full path WITH cwd set -> 'Cannot open data file'), and the
wsprd spot-line parser. A real-decode smoke runs only if a capture yields spots,
and is informational (band-dependent), never a failure.

  python test_wspr.py
"""
import glob
import os
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wspr_live

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def test_iq_to_wav_offset():
    """A tone at dial+1500 Hz must land at ~1500 Hz audio (WSPR window)."""
    fs = 250_000.0
    off = 70_600.0                      # 20m WSPR dial minus a 14025 center
    n = np.arange(int(wspr_live.SLOT_S * fs))
    iq = np.exp(2j * np.pi * (off + 1500.0) / fs * n).astype(np.complex64)
    wav = HERE.parent / "lab" / "_test_wspr.wav"
    wav.parent.mkdir(exist_ok=True)
    wspr_live.iq_to_wav(iq, 14095.6, wav, off_hz=off, fs=fs)
    with wave.open(str(wav), "rb") as w:
        afs = w.getframerate()
        fr = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)
    check("wav is 12 kHz mono ~120 s", afs == 12_000 and abs(len(fr) / afs - 120) < 2,
          f"{len(fr)/afs:.0f}s @ {afs}")
    sp = np.abs(np.fft.rfft(fr * np.hanning(len(fr))))
    peak = np.fft.rfftfreq(len(fr), 1 / afs)[int(np.argmax(sp))]
    check("tone lands in the WSPR window", abs(peak - 1500) < 30, f"peak {peak:.0f} Hz")


def test_wsprd_opens_and_runs():
    """decode_wav must invoke wsprd so it OPENS the file and finishes cleanly
    (no 'Cannot open data file'). Feed a noise wav: expect err=None, 0 spots.
    This is the regression guard for the cwd/basename path handling."""
    if not Path(wspr_live.WSPRD).exists():
        print("  [SKIP] wsprd not installed")
        return
    fs = 12_000
    rng = np.random.default_rng(0)
    fr = (rng.standard_normal(wspr_live.SLOT_S * fs) * 3000).astype(np.int16)
    wav = HERE.parent / "lab" / "_test_wspr_noise.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(fs)
        w.writeframes(fr.tobytes())
    recs, err = wspr_live.decode_wav(wav, 14.0956)
    check("wsprd opens + runs clean on our wav (path handling)", err is None,
          err or f"{len(recs)} spots (0 expected on noise)")


def test_line_parser():
    """The wsprd spot-line regex must extract snr, freq, call, grid."""
    line = " 2358 -21  0.4  14.097091  0  DK6UG JN49 37"
    m = wspr_live._LINE.match(line)
    check("wsprd line matches", m is not None)
    if m:
        msg = m.group(6)
        check("snr parsed", int(m.group(2)) == -21)
        check("freq parsed", abs(float(m.group(4)) - 14.097091) < 1e-6)
        check("callsign extracted", "DK6UG" in wspr_live._CALL.findall(msg))
        g = wspr_live._GRID.search(msg)
        check("grid extracted", bool(g) and g.group(1) == "JN49")


def test_real_smoke_optional():
    """Informational: try a few slots of a long capture; report spots if the band
    was live. Never a failure - WSPR is sparse and CW captures may hold none."""
    import re
    caps = [c for c in glob.glob(str(HERE.parent / "lab" / "**" / "cw_1409*.cs16"),
                                 recursive=True) if os.path.getsize(c) > 60e6]
    caps += [c for c in glob.glob(str(HERE.parent / "lab" / "**" / "cw_14025_*.cs16"),
                                  recursive=True) if os.path.getsize(c) > 60e6]
    if not caps or not Path(wspr_live.WSPRD).exists():
        print("  [SKIP] real-smoke (no long capture and/or wsprd absent)")
        return
    recs, err = wspr_live.slot(caps[0], 8.0)
    print(f"  [INFO] real-smoke: {len(recs) if not err else err} spots "
          f"(0 is fine - band-dependent; plumbing gated above)")


def main():
    print("=" * 60)
    print("WSPR engine-adapter regression gate")
    print("=" * 60)
    test_iq_to_wav_offset()
    test_wsprd_opens_and_runs()
    test_line_parser()
    test_real_smoke_optional()
    print("=" * 60)
    print(f"RESULT: {'ALL PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
