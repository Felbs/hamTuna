"""cw.py - hamTuna: Morse / CW decoder (ham CW + NDB aviation beacons).

The oldest digital mode. On-off keying of a carrier: dit, dah, gaps.
Decoding it turns an IDENTIFIED beacon (we see the carrier) into a
DECODED one (we read its callsign). NDB beacons (190-535 kHz) key their
2-3 letter ID continuously - legal, public, and a clean first target.

Pipeline: mix the carrier to DC -> envelope -> adaptive on/off threshold
-> run-length -> dit/dah/gap classification (self-calibrating WPM) ->
Morse -> text.

Modes:
  selftest - synthesize "NDB" in Morse, add noise, decode it back
  decode   - decode a capture file (cs16) at a given carrier offset

Example:  python cw.py decode --file cap.cs16 --offset -6200
"""
import argparse
import sys
from pathlib import Path

import numpy as np

MORSE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
    "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z", "-----": "0", ".----": "1", "..---": "2",
    "...--": "3", "....-": "4", ".....": "5", "-....": "6", "--...": "7",
    "---..": "8", "----.": "9", "-.-.--": "!", "-..-.": "/"}
INV = {v: k for k, v in MORSE.items()}


def envelope(iq, fs, off_hz, aud=8000):
    from scipy.signal import resample_poly
    from math import gcd
    n = np.arange(len(iq), dtype=np.float64)
    x = iq * np.exp(-2j * np.pi * off_hz / fs * n)
    g = gcd(int(aud), int(fs))
    x = resample_poly(x, int(aud) // g, int(fs) // g).astype(np.complex64)
    env = np.abs(x).astype(np.float32)
    k = max(1, int(aud * 0.008))          # 8 ms smoother
    return np.convolve(env, np.ones(k, np.float32) / k, mode="same"), aud


def decode_env(env, aud):
    """Adaptive on/off -> run lengths -> self-calibrated Morse."""
    hi, lo = np.percentile(env, 90), np.percentile(env, 25)
    if hi - lo < 1e-6:
        return "", {}
    thr = lo + 0.4 * (hi - lo)
    on = env > thr
    # run-length encode
    runs = []
    cur = on[0]
    ln = 1
    for v in on[1:]:
        if v == cur:
            ln += 1
        else:
            runs.append((cur, ln))
            cur, ln = v, 1
    runs.append((cur, ln))
    on_runs = [ln for s, ln in runs if s]
    if len(on_runs) < 3:
        return "", {"runs": len(runs)}
    # dit length = the shorter cluster of ON runs (k-means-lite, 2 groups)
    o = np.array(on_runs, float)
    med = np.median(o)
    dit = np.median(o[o <= med]) or med
    text = []
    sym = ""
    for s, ln in runs:
        if s:                              # tone
            sym += "-" if ln > 2 * dit else "."
        else:                              # gap
            if ln > 5 * dit:
                text.append(MORSE.get(sym, "?") if sym else "")
                text.append(" ")
                sym = ""
            elif ln > 2 * dit:
                if sym:
                    text.append(MORSE.get(sym, "?"))
                    sym = ""
    if sym:
        text.append(MORSE.get(sym, "?"))
    return "".join(text).strip(), {"dit_ms": round(1000 * dit / aud, 1),
                                   "wpm": round(1.2 / (dit / aud), 1),
                                   "elements": len(on_runs)}


def find_offset(iq, fs, search=15000):
    N = 1 << 15
    seg = iq[:len(iq) // N * N].reshape(-1, N) * np.hanning(N).astype(np.float32)
    P = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2).mean(axis=0)
    c = N // 2
    k = int(search / (fs / N))
    band = P[c - k:c + k].copy()
    return (int(np.argmax(band)) - k) * fs / N


def cmd_selftest(args):
    print("=" * 60)
    print("hamTuna CW self-test (synthesize -> noise -> decode)")
    print("=" * 60)
    fs = 250000.0
    aud = fs
    msg = "NDB"
    dit = int(0.06 * fs)                    # ~20 wpm
    seq = []
    for i, ch in enumerate(msg):
        for el in INV[ch]:
            seq.append((1, dit if el == "." else 3 * dit))
            seq.append((0, dit))            # intra-char gap
        seq.append((0, 3 * dit))            # letter gap
    sig = []
    for s, ln in seq:
        sig.append(np.full(ln, float(s)))
    key = np.concatenate(sig)
    t = np.arange(len(key))
    iq = (key * np.exp(2j * np.pi * -6200 / fs * t)).astype(np.complex64)
    rng = np.random.default_rng(1)
    iq += (rng.normal(0, 0.15, len(iq)) + 1j * rng.normal(0, 0.15, len(iq))).astype(np.complex64)
    env, a = envelope(iq, fs, -6200)
    txt, info = decode_env(env, a)
    ok = "NDB" in txt.replace(" ", "")
    print(f"  sent 'NDB' -> decoded '{txt}'  {info}")
    print(f"  {'OK' if ok else 'FAIL'}")
    print("=" * 60)
    print("SELFTEST", "PASS" if ok else "FAIL")
    print("=" * 60)
    return 0 if ok else 1


def cmd_decode(args):
    raw = np.fromfile(args.file, dtype=np.int16).astype(np.float32) / 32768.0
    iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    off = args.offset
    if off is None:
        off = find_offset(iq, args.fs)
        print(f"[cw] auto-found carrier at {off:+.0f} Hz")
    env, a = envelope(iq, args.fs, off)
    txt, info = decode_env(env, a)
    wpm = info.get("wpm", 0)
    print(f"[cw] {info}")
    if not (3 <= wpm <= 45):
        print("[cw] NO READABLE CW - keying rate out of range "
              "(noise, weak signal, or A2A tone-keyed beacon). "
              "Try a longer/stronger capture.")
        return ""
    print(f"[cw] decoded: '{txt}'")
    if txt and txt.replace(" ", "").isalnum():
        print("  -> looks like a real ID! (NDB IDs are 1-3 letters, repeated)")
    return txt


def _ensure_sdr_dll_path():
    """Windows + conda-style python: SoapySDR driver DLLs aren't on PATH
    unless the environment is activated - fix it here so bare launches work."""
    import os
    if os.name != "nt":
        return
    root = Path(sys.executable).resolve().parent
    for p in (root / "Library" / "bin",
              Path(r"C:\Program Files\SDRplay\API\x64"),
              Path(r"C:\Program Files\SDRplay\API")):
        if p.is_dir():
            os.environ["PATH"] = str(p) + os.pathsep + os.environ["PATH"]
            try:
                os.add_dll_directory(str(p))
            except Exception:
                pass


def _open_sdr(antenna, fs=250_000.0):
    _ensure_sdr_dll_path()
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16
    SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
    sdr = SoapySDR.Device("driver=sdrplay")
    sdr.setSampleRate(SOAPY_SDR_RX, 0, fs)
    try:
        sdr.setAntenna(SOAPY_SDR_RX, 0, antenna)
    except Exception:
        pass
    try:
        sdr.setGainMode(SOAPY_SDR_RX, 0, False)
        sdr.setGain(SOAPY_SDR_RX, 0, "IFGR", 30)
        sdr.writeSetting("rfgain_sel", "0")
    except Exception:
        pass
    st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16)
    sdr.activateStream(st)
    return sdr, st


def _grab(sdr, st, secs, fs=250_000.0):
    n_want = int(secs * fs)
    buf = np.empty(2 * 65536, np.int16)
    out = np.empty(2 * n_want, np.int16)
    got = 0
    while got < n_want:
        r = sdr.readStream(st, [buf], 65536, timeoutUs=1_000_000)
        if r.ret > 0:
            n = min(r.ret, n_want - got)
            out[2 * got:2 * (got + n)] = buf[:2 * n]
            got += n
        elif r.ret < 0 and r.ret != -1:
            break
    return ((out[0::2].astype(np.float32) + 1j * out[1::2].astype(np.float32))
            / 32768.0).astype(np.complex64)[:got]


def cmd_listen(args):
    """Live capture on a CW-active frequency, then decode. Real reads are
    logged to lab/cw_decodes.jsonl."""
    import json
    import time as _t
    from SoapySDR import SOAPY_SDR_RX
    sdr, st = _open_sdr(args.antenna, args.fs)
    sdr.setFrequency(SOAPY_SDR_RX, 0, args.khz * 1e3)
    _t.sleep(0.2)
    iq = _grab(sdr, st, args.secs, args.fs)
    sdr.deactivateStream(st); sdr.closeStream(st)
    off = find_offset(iq, args.fs)
    env, a = envelope(iq, args.fs, off)
    txt, info = decode_env(env, a)
    wpm = info.get("wpm", 0)
    if 3 <= wpm <= 45 and txt.strip():
        print(f"[cw] {args.khz} kHz  {info}")
        print(f"[cw] MORSE DECODED: '{txt}'")
        lab = Path(__file__).resolve().parent.parent / "lab"
        lab.mkdir(exist_ok=True)
        rec = {"ts": _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime()),
               "khz": args.khz, "wpm": wpm, "text": txt}
        with open(lab / "cw_decodes.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    else:
        print(f"[cw] {args.khz} kHz: no readable CW (wpm {wpm})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    d = sub.add_parser("decode")
    d.add_argument("--file", required=True)
    d.add_argument("--offset", type=float, default=None)
    d.add_argument("--fs", type=float, default=250000)
    li = sub.add_parser("listen")
    li.add_argument("--khz", type=float, default=14030)   # 20m CW calling area
    li.add_argument("--secs", type=float, default=30)
    li.add_argument("--antenna", default="Antenna C")
    li.add_argument("--fs", type=float, default=250000)
    args = ap.parse_args()
    if args.cmd == "selftest":
        sys.exit(cmd_selftest(args))
    elif args.cmd == "decode":
        cmd_decode(args)
    else:
        cmd_listen(args)


if __name__ == "__main__":
    main()
