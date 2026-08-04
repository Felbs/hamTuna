"""cw_review.py - the human test bench for the CW corpus (:8648).

The decoder says a capture holds Morse; the ear is the referee. This
deck lists every capture the auto-centering selector confirmed as CW,
ranked by how readable its decode looks, and lets you LISTEN: the
envelope at the selected carrier is re-keyed into a clean 600 Hz
sidetone, so you hear the signal the decoder heard - minus the hiss it
had to fight.

Honest ranking (8/03 audit of 304 confirmed-CW captures): median
known-token fraction is 0.00 - most decodes are fragments. 24 captures
clear 30% known tokens, 50 carry a callsign. Those are at the top;
the tail is there so the deck cannot flatter itself.

  python tools/cw_review.py [--port 8648]
File-reader only: never touches the radio, coexists with everything.
"""
import argparse
import glob
import io
import json
import re
import struct
import sys
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cw  # noqa: E402
import cw_center as cc  # noqa: E402

HARVEST = HERE.parent / "lab" / "cw_harvest"
FS = 250000.0  # rate-ok: corpus-file read rate, not a capture path
SIDETONE_HZ = 600.0
WORDS = {"CQ", "DE", "TU", "73", "QRZ", "TEST", "POTA", "SOTA", "RST",
         "TNX", "UP", "QTH", "NAME", "ANT", "PWR", "WX", "HW", "BK", "AR",
         "SK", "K", "R", "ES", "GM", "GA", "GE", "OM", "QSL", "FB", "RIG",
         "KN", "5NN", "599", "DX", "CFM", "PSE", "AGN", "OP", "QRP"}
CALL = re.compile(r"\b[A-Z]{1,2}[0-9][A-Z]{1,3}\b")


def clean_text(t):
    if isinstance(t, (list, tuple)):
        t = " ".join(str(x) for x in t)
    return (t or "").split("{")[0].strip()


def readability(t):
    toks = [x for x in re.split(r"[^A-Z0-9?/]+", t.upper()) if x]
    if not toks:
        return 0.0, []
    known = sum(1 for x in toks if x in WORDS)
    return known / len(toks), CALL.findall(t.upper())


def catalog():
    rows = []
    for j in sorted(glob.glob(str(HARVEST / "cw_*.json"))):
        try:
            d = json.loads(Path(j).read_text())
        except Exception:
            continue
        c = d.get("center") or {}
        if not c.get("is_cw"):
            continue
        t = clean_text(c.get("text"))
        frac, calls = readability(t)
        stem = Path(j).stem
        khz = stem.split("_")[1] if "_" in stem else "?"
        rows.append({"id": stem, "khz": khz,
                     "utc": d.get("trip_utc", ""),
                     "hz": c.get("hz"), "wpm": c.get("wpm"),
                     "rhythm": c.get("rhythm"), "eye": c.get("eye"),
                     "text": t, "known": round(frac, 2), "calls": calls,
                     "iq": d.get("iq_file", "")})
    rows.sort(key=lambda r: (-(r["known"]), -len(r["calls"]),
                             -(r["rhythm"] or 0)))
    return rows


def sidetone_wav(iq_file, hz, secs=12.0, aud_out=8000):
    """Re-key a clean 600 Hz tone with the capture's own envelope."""
    p = HARVEST / iq_file
    iq = cc._load(p, n_max=int(secs * FS))
    env, aud = cw.envelope2(iq, FS, float(hz), aud=aud_out, bw_hz=150)
    runs = cc.key_runs(env, aud)
    key = np.zeros(len(env), np.float32)
    i = 0
    for s, ln in runs:
        if s:
            key[i:i + ln] = 1.0
        i += ln
    # soften edges so it sounds like a radio, not a buzzer
    n_ramp = max(1, int(0.006 * aud))
    key = np.convolve(key, np.hanning(n_ramp) / np.hanning(n_ramp).sum(),
                      "same")
    t = np.arange(len(key)) / aud
    tone = (0.5 * key * np.sin(2 * np.pi * SIDETONE_HZ * t)).astype(np.float32)
    pcm = (np.clip(tone, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(aud))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def keying_strip(iq_file, hz, secs=12.0, cells=600):
    """Compact ON/OFF picture of the keying for the UI."""
    p = HARVEST / iq_file
    iq = cc._load(p, n_max=int(secs * FS))
    env, aud = cw.envelope2(iq, FS, float(hz), aud=8000, bw_hz=150)
    runs = cc.key_runs(env, aud)
    key = np.zeros(len(env), np.uint8)
    i = 0
    for s, ln in runs:
        if s:
            key[i:i + ln] = 1
        i += ln
    n = len(key) // cells or 1
    return [int(x) for x in key[:cells * n].reshape(cells, n).max(axis=1)]


def make_handler():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body if isinstance(body, bytes)
                             else body.encode())

        def do_GET(self):
            from urllib.parse import urlparse, parse_qs
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                self._send(200, (HERE / "cw_review.html").read_bytes(),
                           "text/html")
            elif u.path == "/catalog.json":
                self._send(200, json.dumps({"rows": catalog()}))
            elif u.path == "/audio.wav":
                try:
                    self._send(200, sidetone_wav(q["iq"][0], q["hz"][0]),
                               "audio/wav")
                except Exception as e:
                    self._send(500, str(e), "text/plain")
            elif u.path == "/strip.json":
                try:
                    self._send(200, json.dumps(
                        {"cells": keying_strip(q["iq"][0], q["hz"][0])}))
                except Exception as e:
                    self._send(500, json.dumps({"error": str(e)}))
            else:
                self._send(404, "not found", "text/plain")
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8648)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler())
    rows = catalog()
    print(f"[cw_review] http://127.0.0.1:{a.port}  ({len(rows)} confirmed-CW "
          f"captures, {sum(1 for r in rows if r['known'] >= 0.3)} readable, "
          f"{sum(1 for r in rows if r['calls'])} with callsigns)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
