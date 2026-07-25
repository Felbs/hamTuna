#!/usr/bin/env python3
"""test_ft8.py - regression gate for the FT8 engine-adapter plumbing (#18).

FT8 shipped validated on real captures (13 decodes/slot), but a real signal isn't
a portable test. This gates the parts WE wrote - the offset-correct front end and
the jt9 output parser - with synthetic, self-contained checks (no SDR, no big
capture, no FT8 encoder needed). A real-decode smoke test runs only if a capture
and jt9 happen to be present, and is skipped cleanly otherwise.

  python test_ft8.py
"""
import glob
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ft8_live

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def test_iq_to_wav_offset():
    """iq_to_wav must mix off_hz to baseband so a signal at (off_hz + a) lands at
    audio frequency a in the 12 kHz wav - the whole point of the panel override.
    A tone at center+off+1500 Hz must appear at ~1500 Hz in the wav."""
    fs = 250_000.0
    off = 49_000.0
    a_hz = 1500.0
    n = np.arange(int(15 * fs))
    iq = np.exp(2j * np.pi * (off + a_hz) / fs * n).astype(np.complex64)
    wav = HERE.parent / "lab" / "_test_ft8.wav"
    wav.parent.mkdir(exist_ok=True)
    ft8_live.iq_to_wav(iq, None, wav, off_hz=off, fs=fs)
    with wave.open(str(wav), "rb") as w:
        aud_fs = w.getframerate()
        frames = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)
    check("wav is 12 kHz mono", aud_fs == 12_000, f"got {aud_fs}")
    check("wav holds ~15 s", abs(len(frames) / aud_fs - 15.0) < 1.5,
          f"{len(frames)/aud_fs:.1f}s")
    spec = np.abs(np.fft.rfft(frames * np.hanning(len(frames))))
    freqs = np.fft.rfftfreq(len(frames), 1 / aud_fs)
    peak = freqs[int(np.argmax(spec))]
    check("tone lands at the right audio freq", abs(peak - a_hz) < 30,
          f"peak {peak:.0f} Hz, want {a_hz:.0f}")


def test_offset_wraps_to_usb_window():
    """A tone BELOW the dial (off - a) must NOT alias into the passband as if it
    were above - the USB filter should suppress it. (Sanity that the low-pass +
    real-cast keeps the audio in 0..3.3 kHz.)"""
    fs = 250_000.0
    off = 49_000.0
    n = np.arange(int(4 * fs))
    # a strong tone 6 kHz above the dial is out of the 3.3 kHz USB window
    iq = np.exp(2j * np.pi * (off + 6000.0) / fs * n).astype(np.complex64)
    wav = HERE.parent / "lab" / "_test_ft8b.wav"
    ft8_live.iq_to_wav(iq, None, wav, off_hz=off, fs=fs)
    with wave.open(str(wav), "rb") as w:
        aud = w.getframerate()
        fr = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)
    spec = np.abs(np.fft.rfft(fr * np.hanning(len(fr))))
    freqs = np.fft.rfftfreq(len(fr), 1 / aud)
    inband = spec[freqs <= 3300].max()
    outband = spec[freqs > 4000].max() if (freqs > 4000).any() else 0.0
    check("out-of-window tone is suppressed", inband < 5 * (outband + 1e-9) or inband < 1e3,
          f"inband {inband:.0f} outband {outband:.0f}")


def test_jt9_line_parser():
    """The regex that turns jt9 stdout into structured records must extract snr,
    message, callsigns and grid."""
    line = "000000  -8  0.2 1500 ~  CQ DX W6PAN CM96"
    m = ft8_live._LINE.match(line)
    check("jt9 line matches", m is not None)
    if m:
        msg = m.group(5)
        calls = ft8_live._CALL.findall(msg)
        grid = ft8_live._GRID.search(msg)
        check("snr parsed", int(m.group(2)) == -8)
        check("callsign extracted", "W6PAN" in calls, str(calls))
        check("grid extracted", bool(grid) and grid.group(1) == "CM96",
              grid.group(1) if grid else "none")


def test_real_smoke_optional():
    """If a >=15 s 20 m capture and jt9 are both present, decode one slot and
    require >=1 FT8 record. Skipped cleanly otherwise (portable)."""
    import os
    caps = [c for c in glob.glob(str(HERE.parent / "lab" / "**" / "cw_1402*.cs16"),
                                 recursive=True) if os.path.getsize(c) > 15e6]
    if not caps or not Path(ft8_live.JT9).exists():
        print("  [SKIP] real-decode smoke (no 20m capture and/or jt9 not installed)")
        return
    cap = sorted(caps, key=lambda p: -os.path.getsize(p))[0]
    fs = 250_000.0
    seg = np.fromfile(cap, np.int16, count=2 * int(15 * fs),
                      offset=int(9 * fs) * 4).astype(np.float32) / 32768.0
    iq = (seg[0::2] + 1j * seg[1::2]).astype(np.complex64)
    wav = HERE.parent / "lab" / "_test_ft8_real.wav"
    ft8_live.iq_to_wav(iq, None, wav, off_hz=49_000.0, fs=fs)
    recs, err = ft8_live.decode_wav(wav)
    check("real 20m slot decodes >=1 FT8", (not err) and len(recs) >= 1,
          err or f"{len(recs)} decodes")


def main():
    print("=" * 60)
    print("FT8 engine-adapter regression gate (#18)")
    print("=" * 60)
    test_iq_to_wav_offset()
    test_offset_wraps_to_usb_window()
    test_jt9_line_parser()
    test_real_smoke_optional()
    print("=" * 60)
    print(f"RESULT: {'ALL PASS' if not FAILS else 'FAIL: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
