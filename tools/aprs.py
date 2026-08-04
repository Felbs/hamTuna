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
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parent / "lab"
LAB.mkdir(exist_ok=True)

# skyTuna fusion map: decoded station positions append here (JSONL records
# {t, id, lat, lon, src:"aprs", ...} - sky_panel.py tails the file and
# converts to km offsets server-side). STATION positions only, broadcast
# on-air by the stations themselves; the QTH/receiver location is NEVER
# written to any record.
SKY_APRS = Path(os.environ.get(
    "SKY_APRS_JSONL", r"Z:\src\skyTuna\data\aprs.jsonl"))

FS = 250_000.0  # rate-ok: DEMOD rate only - capture happens at FS_SDR below,
#                 decimated 125/1024 down to FS right after the grab
# CAPTURE AT 2.048M, NEVER 250k (law 8/01): the RSPdx 250 kS/s path delivers
# phase-corrupt, amplitude-suppressed IQ on this box (proven by FM-quieting
# A/B). Capture high, decimate 125/1024 -> FS in software, once, at the top.
FS_SDR = 2_048_000.0
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
    d = {"src": src, "dst": dst, "info": text}
    pos = decode_position(dst, text)
    if pos:
        d.update(pos)
    return d


# ==========================================================================
# APRS position decode (APRS101): MIC-E + plain-text formats
# ==========================================================================
# MIC-E packs latitude into the DESTINATION callsign (one digit per char,
# with N/S, E/W and a +100-deg longitude offset riding the char ranges) and
# longitude/speed/course into the first info bytes. Plain-text positions
# are ddmm.mmN/dddmm.mmW after a '!'/'=' DTI (or '/'/'@' + 7-char time).

_MICE_DTI = "`'\x1c\x1d"          # current/old GPS data type indicators

_PLAIN_POS = re.compile(
    r"(\d{2})([\d ]{2}\.[\d ]{2})([NS])(.)"
    r"(\d{3})([\d ]{2}\.[\d ]{2})([EW])(.)")
_CSE_SPD = re.compile(r"^(\d{3})/(\d{3})")
_ALT_FT = re.compile(r"/A=(\d{6})")
_MICE_ALT = re.compile(r"^[>\]]?([!-{]{3})\}")


def _mice_lat_digit(c):
    """Dest char -> lat digit (K/L/Z = ambiguity space -> 0)."""
    if "0" <= c <= "9":
        return ord(c) - ord("0")
    if "A" <= c <= "J":              # custom message bits
        return ord(c) - ord("A")
    if "P" <= c <= "Y":              # standard message bits
        return ord(c) - ord("P")
    if c in "KLZ":                   # position ambiguity
        return 0
    return None


def decode_mice(dst, info):
    """MIC-E: lat from dest chars, lon/speed/course from info bytes."""
    if len(info) < 9 or info[0] not in _MICE_DTI:
        return None
    base = dst.split("-")[0]
    if len(base) != 6:
        return None
    digs = [_mice_lat_digit(c) for c in base]
    if any(v is None for v in digs):
        return None
    lat_min = digs[2] * 10 + digs[3] + (digs[4] * 10 + digs[5]) / 100.0
    lat = digs[0] * 10 + digs[1] + lat_min / 60.0
    if lat > 90 or lat_min >= 60:
        return None
    north = base[3] >= "P"           # P-Y/Z = North; 0-9/L = South
    offset = base[4] >= "P"          # P-Y/Z = +100 deg longitude
    west = base[5] >= "P"            # P-Y/Z = West; 0-9/L = East
    ld = ord(info[1]) - 28
    if offset:
        ld += 100
    if 180 <= ld <= 189:
        ld -= 80
    elif 190 <= ld <= 199:
        ld -= 190
    lm = ord(info[2]) - 28
    if lm >= 60:
        lm -= 60
    lh = ord(info[3]) - 28
    if not (0 <= ld <= 179 and 0 <= lm <= 59 and 0 <= lh <= 99):
        return None
    lon = ld + (lm + lh / 100.0) / 60.0
    sp = ord(info[4]) - 28
    dc = ord(info[5]) - 28
    se = ord(info[6]) - 28
    if not (0 <= sp <= 99 and 0 <= dc <= 99 and 0 <= se <= 99):
        return None
    speed = sp * 10 + dc // 10
    course = (dc % 10) * 100 + se
    if speed >= 800:
        speed -= 800
    if course >= 400:
        course -= 400
    out = {"lat": round(lat if north else -lat, 5),
           "lon": round(-lon if west else lon, 5),
           "speed_kt": speed, "course": course, "fmt": "mice"}
    tail = info[9:]                  # optional status; may lead with alt
    m = _MICE_ALT.match(tail)
    if m:
        a = m.group(1)
        out["alt_m"] = ((ord(a[0]) - 33) * 91 * 91 + (ord(a[1]) - 33) * 91
                        + (ord(a[2]) - 33)) - 10000
    return out


def decode_plain(info):
    """'!'/'=' uncompressed ddmm.mmN/dddmm.mmW, '/'/'@' + 7-char time."""
    if not info:
        return None
    dti = info[0]
    if dti in "!=":
        body = info[1:]
    elif dti in "/@" and len(info) > 8:
        body = info[8:]              # skip DDHHMMz / HHMMSSh timestamp
    else:
        return None
    m = _PLAIN_POS.search(body)
    if not m:
        return None
    lat = int(m.group(1)) + float(m.group(2).replace(" ", "0")) / 60.0
    if m.group(3) == "S":
        lat = -lat
    lon = int(m.group(5)) + float(m.group(6).replace(" ", "0")) / 60.0
    if m.group(7) == "W":
        lon = -lon
    if abs(lat) > 90 or abs(lon) > 180:
        return None
    out = {"lat": round(lat, 5), "lon": round(lon, 5), "fmt": "plain"}
    rest = body[m.end():]
    mc = _CSE_SPD.match(rest)
    if mc:
        out["course"] = int(mc.group(1))
        out["speed_kt"] = int(mc.group(2))
        rest = rest[mc.end():]
    ma = _ALT_FT.search(rest)
    if ma:
        out["alt_m"] = round(int(ma.group(1)) * 0.3048, 1)
    rest = rest.strip()
    if rest:
        out["comment"] = rest[:40]
    return out


def decode_position(dst, info):
    """Best-effort APRS position from an AX.25 frame. None if positionless."""
    if info and info[0] in _MICE_DTI:
        return decode_mice(dst, info)
    return decode_plain(info)


def mice_encode(lat, lon, speed_kt=0, course=0, symbol=">", table="/"):
    """Known coords -> (dest, info) MIC-E pair. Selftest-side inverse of
    decode_mice (standard message bits, no ambiguity)."""
    north, lat = lat >= 0, abs(lat)
    west, lon = lon <= 0, abs(lon)
    latmin = (lat - int(lat)) * 60.0
    digits = f"{int(lat):02d}{int(latmin):02d}{round((latmin % 1) * 100):02d}"
    ld = int(lon)
    lm_f = (lon - ld) * 60.0
    lm, lh = int(lm_f), round((lm_f - int(lm_f)) * 100)
    offset = ld <= 9 or ld >= 100
    dst = ""
    for i, ch in enumerate(digits):
        dig = int(ch)
        up = (i < 3                              # standard msg bits
              or (i == 3 and north)
              or (i == 4 and offset)
              or (i == 5 and west))
        dst += chr((ord("P") if up else ord("0")) + dig)
    if ld <= 9:
        c_d = ld + 118                           # (ld+190) - 100 + 28
    elif ld <= 99:
        c_d = ld + 28
    elif ld <= 109:
        c_d = ld + 8                             # (ld-100+80) + 28
    else:
        c_d = ld - 72                            # (ld-100) + 28
    c_m = lm + 88 if lm <= 9 else lm + 28
    sp, rem = int(speed_kt) // 10, int(speed_kt) % 10
    dc = rem * 10 + int(course) // 100
    se = int(course) % 100
    info = "`" + "".join(chr(c + 28) for c in
                         (c_d - 28, c_m - 28, lh, sp, dc, se)) + symbol + table
    return dst, info


# ==========================================================================
# skyTuna map emitter
# ==========================================================================
def emit_positions(frames, path=SKY_APRS, t=None):
    """Append position-bearing frames to the sky panel's aprs.jsonl.
    Schema (matches sky_panel LAYERS/TRACK_FIELDS): t, id, lat, lon,
    src:"aprs" + optional speed_kt/course/alt_m/comment. Never the QTH."""
    recs, seen = [], set()
    for f in frames:
        if f.get("lat") is None or f.get("lon") is None:
            continue
        key = (f["src"], f["lat"], f["lon"])
        if key in seen:
            continue
        seen.add(key)
        rec = {"t": round(t if t is not None else time.time(), 2),
               "id": f["src"], "lat": f["lat"], "lon": f["lon"],
               "src": "aprs"}
        for k in ("speed_kt", "course", "alt_m", "comment"):
            if f.get(k) is not None:
                rec[k] = f[k]
        recs.append(rec)
    if recs:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
    return len(recs)


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

    # [3] plain-text position decode off the same decoded frame
    want = (38 + 52.30 / 60.0, -(77 + 2.00 / 60.0))
    pf = next((f for f in frames if f.get("lat") is not None), None)
    good = (pf is not None
            and abs(pf["lat"] - want[0]) < 0.01
            and abs(pf["lon"] - want[1]) < 0.01)
    print(f"[3] uncompressed position decode: "
          f"{'OK' if good else 'FAIL'}"
          + (f"  ({pf['lat']:.5f},{pf['lon']:.5f})" if pf else "  (no pos)"))
    ok &= good

    # [4] timestamped '@' variant (parser-level; same position grammar)
    tp = decode_position("APRS", "@092345z3852.30N/07702.00W>on time")
    good = (tp is not None and abs(tp["lat"] - want[0]) < 0.01
            and abs(tp["lon"] - want[1]) < 0.01)
    print(f"[4] timestamped '@' position decode: {'OK' if good else 'FAIL'}")
    ok &= good

    # [5] MIC-E roundtrip through the full RF chain: known coords -> dest
    # digits + info bytes -> AX.25 -> AFSK -> FM -> decode -> coords
    m_lat, m_lon, m_spd, m_cse = 33.42733, -112.12417, 23, 251
    dst, minfo = mice_encode(m_lat, m_lon, m_spd, m_cse)
    frames = demod(synth_iq(build_ax25("N0CALL-7", dst, minfo)))
    mf = next((f for f in frames if f.get("fmt") == "mice"), None)
    good = (mf is not None
            and abs(mf["lat"] - m_lat) < 0.01
            and abs(mf["lon"] - m_lon) < 0.01
            and mf["speed_kt"] == m_spd and mf["course"] == m_cse)
    print(f"[5] MIC-E roundtrip (dest={dst}): {'OK' if good else 'FAIL'}"
          + (f"  ({mf['lat']:.5f},{mf['lon']:.5f} "
             f"{mf['speed_kt']}kt/{mf['course']}deg)" if mf else "  (no fix)"))
    ok &= good

    # [6] map emitter schema (temp file - never the real map feed here)
    tmp = LAB / "aprs_emit_selftest.jsonl"
    tmp.unlink(missing_ok=True)
    n = emit_positions([{"src": "N0CALL-7", "lat": 33.42733,
                         "lon": -112.12417, "speed_kt": 23, "course": 251}],
                       path=tmp, t=1234.0)
    rec = json.loads(tmp.read_text().strip()) if n else {}
    tmp.unlink(missing_ok=True)
    good = (n == 1 and rec.get("id") == "N0CALL-7"
            and rec.get("src") == "aprs" and rec.get("t") == 1234.0
            and rec.get("lat") == 33.42733 and rec.get("lon") == -112.12417)
    print(f"[6] map emitter JSONL schema: {'OK' if good else 'FAIL'}")
    ok &= good
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
    sdr.setSampleRate(SOAPY_SDR_RX, 0, FS_SDR)
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
    n_want = int(args.secs * FS_SDR)
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
    # decimate 2.048M -> 250k (exact 125/1024); every downstream sample-rate
    # assumption (FS) is unchanged - the rate conversion happens once, here.
    from scipy.signal import resample_poly
    iq = resample_poly(iq, 125, 1024).astype(np.complex64)
    print(f"[capture] {len(iq)/FS:.1f}s captured, demodulating ...")
    # H6 control: RF burst counter - separates "band is quiet" from "our
    # demod is deaf". 10 ms envelope cells; a burst = >=80 ms above 2.5x
    # the noise floor (APRS packets run ~300-1000 ms).
    k = int(0.01 * FS)
    env = np.abs(iq[:len(iq) // k * k]).reshape(-1, k).mean(axis=1)
    floor = float(np.median(env)) + 1e-9
    hot = env > 2.5 * floor
    bursts = 0
    run = 0
    for h in hot:
        run = run + 1 if h else 0
        if run == 8:
            bursts += 1
    frames = demod(iq)
    print(f"[control] RF bursts >=80ms: {bursts}  (bursts>>frames = demod "
          f"deficit; ~0 = quiet band)")
    print(f"[result] CRC-valid AX.25 frames: {len(frames)}")
    seen = {}
    for f in frames:
        seen.setdefault(f["src"], f)
    for src, f in seen.items():
        pos = (f"  [{f['lat']:.5f},{f['lon']:.5f}]"
               if f.get("lat") is not None else "")
        print(f"    {src:<10} > {f['dst']:<8} {f['info'][:64]}{pos}")
    if not frames:
        print("    (none this window - APRS is bursty; try --secs 120+)")
    n_emit = emit_positions(frames)
    if n_emit:
        print(f"[map] {n_emit} position(s) -> {SKY_APRS} (sky panel :8644)")


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
