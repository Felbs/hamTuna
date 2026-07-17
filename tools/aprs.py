"""aprs.py - hamTuna campaign 1: APRS position beacons on 144.390 MHz.

APRS is amateur radio's AIS: hams beacon their callsign, position, and
status over AX.25 packets - AFSK 1200 baud (Bell 202 tones, 1200/2200 Hz)
on FM. Same HDLC framing and CRC-16/X.25 truth dial we field-proved on
the Potomac's AIS buoys; only the modem underneath is new.

Pipeline: IQ @ 250k -> NBFM discriminator -> audio 48k -> dual tone
envelopes (1200/2200) -> soft bits @ 1200 bd -> NRZI -> HDLC destuff ->
CRC-16 gate -> AX.25 addresses (callsigns!) + APRS info text.

Modes:
  selftest - full synthetic roundtrip (AX.25 -> AFSK -> FM -> decode)
  capture  - N seconds live on 144.390, station table

Example:  python aprs.py capture --secs 60 --antenna "Antenna A"
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parent / "lab"
LAB.mkdir(exist_ok=True)

FS = 250_000.0
FREQ = 144.390e6
AUD = 48_000.0
BAUD = 1200.0
MARK, SPACE = 1200.0, 2200.0


def _ensure_sdr_dll_path():
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


_ensure_sdr_dll_path()


# ==========================================================================
# shared HDLC/CRC plumbing (the AIS-proven versions)
# ==========================================================================
def crc16_x25(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc ^ 0xFFFF


def stuff(bits):
    out, run = [], 0
    for b in bits:
        out.append(b)
        run = run + 1 if b == 1 else 0
        if run == 5:
            out.append(0)
            run = 0
    return out


def destuff(bits):
    out, run, i = [], 0, 0
    while i < len(bits):
        b = bits[i]
        out.append(b)
        run = run + 1 if b == 1 else 0
        if run == 5:
            i += 1
            if i < len(bits) and bits[i] == 1:
                return None
            run = 0
        i += 1
    return out


def nrzi_encode(bits):
    out, cur = [], 0
    for b in bits:
        if b == 0:
            cur ^= 1
        out.append(cur)
    return out


def nrzi_decode(line):
    out = np.empty(len(line) - 1, np.int8)
    for i in range(1, len(line)):
        out[i - 1] = 1 if line[i] == line[i - 1] else 0
    return out


def find_frames(bits, min_bits=136, max_bits=3000):
    s = "".join(str(int(b)) for b in bits)
    flag = "01111110"
    idx = [i for i in range(len(s) - 8) if s[i:i + 8] == flag]
    hits = []
    for a_i in range(len(idx)):
        for b_i in range(a_i + 1, min(a_i + 30, len(idx))):
            a, b = idx[a_i] + 8, idx[b_i]
            if not (min_bits <= b - a <= max_bits):
                continue
            raw = destuff([int(c) for c in s[a:b]])
            if raw is None or len(raw) % 8 != 0 or len(raw) < 17 * 8:
                continue
            by = bytes(sum(bit << k for k, bit in enumerate(raw[j:j + 8]))
                       for j in range(0, len(raw), 8))   # LSB-first wire
            body, fcs = by[:-2], by[-2] | (by[-1] << 8)
            if crc16_x25(body) == fcs:
                hits.append(body)
    return hits


# ==========================================================================
# AX.25 parse
# ==========================================================================
def parse_ax25(body):
    if len(body) < 16:
        return None
    def call(seg):
        cs = "".join(chr((c >> 1) & 0x7F) for c in seg[:6]).strip()
        ssid = (seg[6] >> 1) & 0x0F
        return f"{cs}-{ssid}" if ssid else cs
    dst = call(body[0:7])
    src = call(body[7:14])
    i = 14
    while i + 7 <= len(body) and not (body[i - 1] & 0x01):   # digipeaters
        i += 7
    if i + 2 > len(body):
        return None
    info = body[i + 2:]
    try:
        text = info.decode("ascii", errors="replace")
    except Exception:
        text = repr(info)
    return {"src": src, "dst": dst, "info": text}


# ==========================================================================
# AFSK demod
# ==========================================================================
def afsk_softbits(audio, fs=AUD):
    """Dual sliding tone envelopes -> soft bits at 1200 bd."""
    n = np.arange(len(audio))
    spb = fs / BAUD
    w = int(spb)
    box = np.ones(w, np.float32) / w
    e = {}
    for name, f in (("mark", MARK), ("space", SPACE)):
        z = audio * np.exp(-2j * np.pi * f / fs * n)
        # low-pass the complex product FIRST, then magnitude (non-coherent
        # tone detector); |z| before filtering would just be |audio|
        e[name] = np.abs(np.convolve(z, box, mode="same")).astype(np.float32)
    d = e["mark"] - e["space"]
    # integrate-and-dump at the bit rate with a simple zero-crossing nudge
    nb = int(len(d) / spb) - 2
    soft = np.empty(nb, np.float32)
    pos = 0.0
    for k in range(nb):
        p = int(pos)
        if p + w >= len(d):
            soft = soft[:k]
            break
        soft[k] = float(np.mean(d[p:p + w]))
        # nudge: align to the strongest local transition
        pos += spb
    return soft


def demod(iq, fs=FS):
    from scipy.signal import resample_poly
    from math import gcd
    iq = iq - np.mean(iq)
    disc = np.angle(iq[1:] * np.conj(iq[:-1])).astype(np.float32)
    g = gcd(int(AUD), int(fs))
    audio = resample_poly(disc, int(AUD) // g, int(fs) // g).astype(np.float32)
    audio -= float(np.mean(audio))
    soft = afsk_softbits(audio)
    frames = []
    for sgn in (1.0, -1.0):
        line = (soft * sgn > 0).astype(np.int8)
        bits = nrzi_decode(line)
        for body in find_frames(bits):
            d = parse_ax25(body)
            if d:
                frames.append(d)
        if frames:
            break
    return frames


# ==========================================================================
# selftest: AX.25 -> AFSK -> NBFM -> decode
# ==========================================================================
def build_ax25(src, dst, info):
    def enc_call(cs, last=False):
        base, _, ssid = cs.partition("-")
        b = bytearray((ord(c) << 1) for c in base.ljust(6))
        b.append(((int(ssid or 0) & 0x0F) << 1) | 0x60 | (1 if last else 0))
        return bytes(b)
    body = enc_call(dst) + enc_call(src, last=True) + b"\x03\xf0" + info.encode()
    fcs = crc16_x25(body)
    frame = body + bytes([fcs & 0xFF, fcs >> 8])
    fb = []
    for byte in frame:
        for i in range(8):
            fb.append((byte >> i) & 1)
    return [0] * 8 + [0, 1, 1, 1, 1, 1, 1, 0] + stuff(fb) + [0, 1, 1, 1, 1, 1, 1, 0] + [0] * 8


def synth_iq(wire_bits, fs=FS, noise=0.03):
    line = nrzi_encode(wire_bits)
    spb_a = AUD / BAUD
    audio = np.zeros(int(len(line) * spb_a) + 100, np.float32)
    phase = 0.0
    for i, b in enumerate(line):
        f = MARK if b else SPACE
        a, z = int(i * spb_a), int((i + 1) * spb_a)
        t = np.arange(z - a)
        audio[a:z] = np.sin(phase + 2 * np.pi * f / AUD * t)
        phase += 2 * np.pi * f / AUD * (z - a)
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(fs), int(AUD))
    aud_up = resample_poly(audio, int(fs) // g, int(AUD) // g)
    dev = 3000.0
    ph = np.cumsum(2 * np.pi * dev * aud_up / fs)
    iq = 0.5 * np.exp(1j * ph).astype(np.complex64)
    rng = np.random.default_rng(3)
    iq += (rng.normal(0, noise, len(iq)) + 1j * rng.normal(0, noise, len(iq))
           ).astype(np.complex64)
    return iq


def cmd_selftest(args):
    print("=" * 62)
    print("hamTuna APRS self-test (AX.25 -> AFSK1200 -> NBFM -> decode)")
    print("=" * 62)
    ok = True
    c = crc16_x25(b"123456789")
    print(f"[1] CRC-16/X.25 check value: {c:04X}  {'OK' if c == 0x906E else 'FAIL'}")
    ok &= (c == 0x906E)
    wire = build_ax25("N0CALL-9", "APRS",
                      "!3852.30N/07702.00W>hamTuna selftest")
    iq = synth_iq(wire)
    frames = demod(iq)
    hit = any(f["src"] == "N0CALL-9" and "3852.30N" in f["info"] for f in frames)
    print(f"[2] synthetic beacon roundtrip: decoded={len(frames)}  "
          f"callsign+position match={'OK' if hit else 'FAIL'}")
    for f in frames[:2]:
        print(f"    {f['src']} > {f['dst']}: {f['info'][:60]}")
    ok &= hit
    print("=" * 62)
    print("SELFTEST", "PASS" if ok else "FAIL")
    print("=" * 62)
    return 0 if ok else 1


# ==========================================================================
# live capture
# ==========================================================================
def cmd_capture(args):
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16
    SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
    sdr = SoapySDR.Device("driver=sdrplay")
    sdr.setSampleRate(SOAPY_SDR_RX, 0, FS)
    sdr.setFrequency(SOAPY_SDR_RX, 0, FREQ)
    try:
        sdr.setAntenna(SOAPY_SDR_RX, 0, args.antenna)
    except Exception:
        pass
    try:
        sdr.setGainMode(SOAPY_SDR_RX, 0, False)
        sdr.setGain(SOAPY_SDR_RX, 0, "IFGR", 22)
        sdr.writeSetting("rfgain_sel", "0")
    except Exception:
        pass
    st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16)
    sdr.activateStream(st)
    print(f"[capture] {args.secs:.0f}s @ 144.390 MHz on {args.antenna} "
          f"(APRS beacons are bursty - longer is better)")
    n_want = int(args.secs * FS)
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
    sdr.deactivateStream(st)
    sdr.closeStream(st)
    iq = ((out[0::2].astype(np.float32) + 1j * out[1::2].astype(np.float32))
          / 32768.0).astype(np.complex64)[:got]
    print(f"[capture] {len(iq)/FS:.1f}s captured, demodulating ...")
    frames = demod(iq)
    print(f"[result] CRC-valid AX.25 frames: {len(frames)}")
    seen = {}
    for f in frames:
        seen.setdefault(f["src"], f)
    for src, f in seen.items():
        print(f"    {src:<10} > {f['dst']:<8} {f['info'][:64]}")
    if not frames:
        print("    (none this window - APRS is bursty; try --secs 120+)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    c = sub.add_parser("capture")
    c.add_argument("--secs", type=float, default=60)
    c.add_argument("--antenna", default="Antenna A")
    args = ap.parse_args()
    if args.cmd == "selftest":
        sys.exit(cmd_selftest(args))
    elif args.cmd == "capture":
        cmd_capture(args)


if __name__ == "__main__":
    main()
