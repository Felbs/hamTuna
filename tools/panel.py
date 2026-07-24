#!/usr/bin/env python3
"""panel.py - hamTuna adaptive SDR panel (SDRuno-style, in the browser).

The Tuna thesis, made visible: a real spectrum + waterfall receiver like
SDRuno/SDRplay, but with a TRUTH DIAL — the software surfaces how well the
active mode is decoding and closes the loop (auto-find the signal, self-
calibrate, show confidence). Every mode plugs into one registry so "add a
ham mode" == "add a decoder function".

  python tools/panel.py            # http://localhost:8647

v2: OLED-tuned waterfall (true-black floor -> white-hot peaks), click-to-peak
navigation, live rolling Morse transcript, and a live CW audio stream you can
listen to while you read. Single SDR via radio_lock@80, Antenna C (HF).
"""
import argparse
import json
import os
import re
import struct
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, r"Z:\src\gr-radiotuna\tools")
import cw
import cw_lm
import cw_quality
import hamdb
try:
    import radio_lock
except Exception:
    radio_lock = None

FS = 250_000.0
N_FFT = 2048
N_HI = 8192            # high-res spectrum bins (~30 Hz/bin over 250 kHz) for zoom
DISP_BINS = 500
DECODE_SECS = 24         # COHERENT decode window. Short windows (v2 used 6 s)
DECODE_EVERY = 4         # chop transmissions mid-word/char -> fragmented stubs.
SPAN_KHZ = FS / 1e3
AUD_DEC = 31             # 250000/31 = 8064.5 Hz audio
AUD_FS = int(FS / AUD_DEC)
BFO_HZ = 600.0          # CW carrier is mixed to this pitch

BANDS = {"160m": 1830, "80m": 3560, "40m": 7030, "30m": 10120, "20m": 14030,
         "17m": 18080, "15m": 21030, "12m": 24906, "10m": 28030}
# CW lives at the bottom of each band (ham band plan). Signal-hunt + auto-tune
# stay inside these so they lock CW, not the FT8/SSB above.
CW_SUB = {"160m": (1800, 1843), "80m": (3500, 3600), "40m": (7000, 7040),
          "30m": (10100, 10130), "20m": (14000, 14070), "17m": (18068, 18095),
          "15m": (21000, 21070), "12m": (24890, 24915), "10m": (28000, 28070)}
MODES = ["CW", "SSB", "AM", "FM", "APRS", "FT8"]


def detect_signals():
    """Carriers (peaks over noise) inside the current band's CW sub-band and the
    visible window — the CW 'channels' on the air right now. This is navigation."""
    with _lock:
        db = np.array(SPEC["db"]); c = STATE["center_khz"]
        noise = SPEC["noise_db"]; band = STATE["band"]
    if not len(db):
        return []
    sub = CW_SUB.get(band)
    lo_khz = c - SPAN_KHZ / 2
    binkhz = SPAN_KHZ / len(db)
    thr = noise + 7.0
    peaks = []
    for i in range(2, len(db) - 2):
        f = lo_khz + i * binkhz
        if sub and not (sub[0] <= f <= sub[1]):
            continue
        v = db[i]
        if v > thr and v >= db[i - 1] and v > db[i + 1] and v >= db[i - 2] and v > db[i + 2]:
            peaks.append((round(f, 2), round(v - noise, 1)))
    peaks.sort()
    merged = []
    for f, s in peaks:                 # merge carriers within 0.4 kHz
        if merged and f - merged[-1][0] < 0.4:
            if s > merged[-1][1]:
                merged[-1] = (f, s)
        else:
            merged.append((f, s))
    return [{"khz": f, "snr": s} for f, s in merged]

STATE = {"center_khz": 14030.0, "tune_khz": 14030.0, "band": "20m", "mode": "CW",
         "ifgr": 30, "rfsel": 0, "running": True, "antenna": "Antenna C",
         "lock": "none", "err": "", "chlock": False, "lock_off": 0.0, "last_off": 0.0}
# center_khz = the SDR/display center (the window); tune_khz = the CURSOR (the
# exact freq we decode/listen to, movable within the window, SDRuno-style).


def cur_off_hz():
    return (STATE["tune_khz"] - STATE["center_khz"]) * 1000.0
SPEC = {"db": [0.0] * DISP_BINS, "peak_db": -120.0, "noise_db": -120.0, "ts": 0.0,
        "hi": None}          # hi = high-res np array for zoom (kept in memory, not JSON)
DECODE = {"text": "", "wpm": 0.0, "q": 0.0, "conf": 0.0, "elements": 0,
          "mode": "CW", "ts": 0.0, "hint": "", "offset_hz": 0.0,
          "eye_q": 0.0, "eye_db": 0.0, "copy_pct": 0, "verdict": "—", "route": "classic"}
TRANSCRIPT = deque(maxlen=60)
AUDIO = deque(maxlen=AUD_FS * 4)
SIGLIST = {"band": None, "sigs": []}   # classifier's authoritative carrier list: [{khz,snr,cw,wpm}]
_lock = threading.Lock()
_alock = threading.Lock()
_win = np.hanning(N_FFT).astype(np.float32)
_win_hi = np.hanning(N_HI).astype(np.float32)
_lp = firwin(159, 1500.0 / (FS / 2)).astype(np.float32)   # audio CW filter
_narrow8k = firwin(129, 400.0 / 4000.0).astype(np.float32)  # ±400 Hz single-station filter @8 kHz


def envelope_locked(iq, off_hz, aud=8000):
    """Isolate ONE station: shift its carrier to DC, keep only ±400 Hz (rejects
    adjacent CW), then envelope — so the cursor copies just that one signal."""
    from math import gcd
    from scipy.signal import resample_poly
    n = np.arange(len(iq), dtype=np.float64)
    x = (iq * np.exp(-2j * np.pi * off_hz / FS * n)).astype(np.complex64)
    g = gcd(int(aud), int(FS))
    xr = resample_poly(x, int(aud) // g, int(FS) // g)      # complex -> 8 kHz
    xf = lfilter(_narrow8k, 1.0, xr)                        # tight ±250 Hz
    env = np.abs(xf).astype(np.float32)
    k = max(1, int(aud * 0.008))
    return np.convolve(env, np.ones(k, np.float32) / k, mode="same"), aud

# IQ ring buffer: the reader writes here fast; the decoder snapshots a long
# coherent window off-thread so a slow decode never stalls the SDR read
# (a stalled read drops samples -> gapped timing -> real gibberish).
RING = np.zeros(int(30 * FS), np.complex64)
_rw = 0
_rfill = 0
_rlock = threading.Lock()


def ring_write(iq):
    global _rw, _rfill
    m = len(iq); L = len(RING)
    with _rlock:
        if _rw + m <= L:
            RING[_rw:_rw + m] = iq
        else:
            k = L - _rw; RING[_rw:] = iq[:k]; RING[:m - k] = iq[k:]
        _rw = (_rw + m) % L
        _rfill = min(L, _rfill + m)


def ring_snapshot(secs):
    n = min(int(secs * FS), _rfill)
    if n < FS:
        return None
    with _rlock:
        idx = (np.arange(_rw - n, _rw) % len(RING))
        return RING[idx].astype(np.complex64)


def ring_clear():
    global _rfill
    with _rlock:
        _rfill = 0


def _spectrum(iq):
    """High-res full-band spectrum (N_HI bins ~30 Hz/bin) so the UI can ZOOM into
    it with real resolution. Same total FFT work as the old 2048/32-seg overview."""
    n = len(iq) // N_HI * N_HI
    if n < N_HI:
        return None
    seg = iq[:n].reshape(-1, N_HI) * _win_hi
    p = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2).mean(0)
    return (10 * np.log10(p + 1e-9)).astype(np.float32)          # length N_HI


def _downsample(db, bins):
    """Max-pool a db array to `bins` (keeps peaks visible when zoomed out)."""
    if len(db) <= bins:
        return db
    step = len(db) // bins
    return db[:step * bins].reshape(bins, step).max(1)


def _view_slice(hi, center, vc, vs, bins=DISP_BINS):
    """Slice the high-res spectrum to [vc±vs/2], pool to `bins`, and return the
    slice with the EXACT frequency bounds it represents (bin-aligned) so the UI maps
    the signal to the right pixel — otherwise sub-bin rounding makes it drift on zoom.
    Returns (db_list, actual_center_khz, actual_span_khz)."""
    lo_khz = center - SPAN_KHZ / 2
    binkhz = SPAN_KHZ / len(hi)
    i0 = int(round((vc - vs / 2 - lo_khz) / binkhz))
    i1 = int(round((vc + vs / 2 - lo_khz) / binkhz))
    i0 = max(0, min(len(hi) - 2, i0)); i1 = max(i0 + 1, min(len(hi), i1))
    seg = hi[i0:i1]
    if len(seg) > bins:                       # pooled: covers exactly step*bins input bins
        step = len(seg) // bins
        seg = seg[:step * bins].reshape(bins, step).max(1)
        used = step * bins
    else:
        used = len(seg)
    act_lo = lo_khz + i0 * binkhz
    act_hi = lo_khz + (i0 + used) * binkhz
    return seg, (act_lo + act_hi) / 2, act_hi - act_lo


_AI = "unset"          # lazy-loaded neural decoder: (model, torch) | None


def _get_ai():
    """Load the trained CNN-BiLSTM-CTC model once. Returns (model, torch) or None
    if torch/model unavailable - so the panel runs fine without the AI."""
    global _AI
    if _AI == "unset":
        try:
            import cw_ai
            if cw_ai.MODEL_PATH.exists():
                _AI = cw_ai.load_model()          # (model, torch)
            else:
                _AI = None
        except Exception:
            _AI = None
    return _AI


def decode_cw(iq):
    # base = where to look: the locked carrier, or the live cursor. EITHER way
    # snap ±400 Hz to the actual carrier there (a bin-resolution cursor or a
    # slightly-off lock is still grabbed), then narrow-filter just that signal.
    base = STATE["lock_off"] if STATE["chlock"] else cur_off_hz()
    s = iq[:int(FS)] if len(iq) > FS else iq
    n = np.arange(len(s))
    x = (s * np.exp(-2j * np.pi * base / FS * n)).astype(np.complex64)
    off = base + cw.find_offset(x, FS, 700)   # find the carrier within +/-700 Hz of the
    #                                           cursor (was 400) so a slightly-off click
    #                                           still lands the decode ON the signal you hear
    if not STATE["chlock"]:
        STATE["last_off"] = off
    env, aud = envelope_locked(iq, off)   # narrow — decode just that one signal
    # CLASSIC apparatus-routed decoder is PRIMARY: on held-out real captures it
    # reads verified callsigns 13/13 while the current neural model (overfit to its
    # 2-capture validation set) reads 0/13. The neural path is opt-in (STATE['use_ai'])
    # until it actually beats classic on the callsign-recall metric.
    txt, info = cw.decode_env_auto(env, aud)
    route = info.get("route", "classic")
    if STATE.get("use_ai"):
        ai = _get_ai()
        if ai is not None:
            try:
                import cw_ai
                wenv, waud = cw.envelope(iq, FS, off)
                ai_txt = cw_ai.decode_feat(ai[0], cw_ai.env_to_feat(wenv, waud), ai[1])
                if ai_txt and len([c for c in ai_txt if c != " "]) >= 3:
                    txt, route = ai_txt, "neural"
            except Exception:
                pass
    txt = cw_lm.rescore(txt)                          # ham LM: re-segment words + repair '?'
    chars = [c for c in txt if c != " "]
    q = round(sum(1 for c in chars if c != "?") / len(chars), 3) if chars else 0.0
    wpm = info.get("wpm", 0.0)
    ok = 3 <= wpm <= 45 and txt.strip()
    conf = round(q * min(1.0, len(chars) / 10), 3) if ok else 0.0
    # HONEST quality: eye-opening Q (measured pre-decode, can't be faked by a
    # confident-but-wrong decode the way q_ratio can) + a plain verdict.
    eye_q, eye_db, copy_pct, _, _, _ = cw_quality.eye_opening(env)
    if eye_q >= cw_quality.Q_SOLID:
        verdict = "SOLID"
    elif eye_q >= cw_quality.Q_READABLE:
        verdict = "READABLE"
    else:
        verdict = "FAILING"
    if eye_q < cw_quality.Q_READABLE:
        hint = ("eye is CLOSED — signal is fading/smeared, not just weak; "
                "the ✓CW list shows which carriers are actually copyable")
    elif not ok:
        hint = "no readable CW — click a ✓CW signal on the list or Auto-Tune"
    else:
        hint = ""
    return {"text": txt if ok else "", "wpm": wpm, "q": q, "conf": conf,
            "eye_q": round(eye_q, 2), "eye_db": round(eye_db, 1),
            "copy_pct": round(copy_pct), "verdict": verdict,
            "route": route,
            "elements": info.get("elements", 0), "offset_hz": round(off, 1), "hint": hint}


DECODERS = {"CW": decode_cw}

# ── logbook: harvest callsigns like a real ham, and score them ──
LOGFILE = HERE.parent / "lab" / "cw_log.jsonl"
HARVEST_DIR = HERE.parent / "lab" / "cw_harvest"
LOGBOOK = {}                       # call -> record
CALL_RE = re.compile(r"^[A-Z0-9]{1,2}[0-9][A-Z]{1,4}$")
PROSIGN = {"CQ", "DE", "QRL", "QSL", "QSO", "QTH", "QRZ", "QRM", "QRN", "QSB",
           "QRP", "TU", "GM", "GA", "GE", "RST", "AGN", "BK", "AR", "SK", "KN",
           "73", "88", "FB", "OM", "UR", "PSE", "POTA", "SOTA", "WX", "TNX"}


def extract_calls(text):
    """Callsign-pattern tokens that are confident: repeated (hams send calls
    2-3x) or right after DE/CQ. Confidence-gating keeps decode noise out."""
    toks = text.upper().split()
    out = {}
    for i, t in enumerate(toks):
        if t in PROSIGN or not CALL_RE.match(t):
            continue
        conf = 0
        if toks.count(t) >= 2:
            conf += 2
        if i > 0 and toks[i - 1] in ("DE", "CQ"):
            conf += 2
        if 3 <= len(t) <= 6:
            conf += 1
        if conf >= 2:
            out[t] = max(out.get(t, 0), conf)
    return out


def _load_log():
    try:
        for line in open(LOGFILE, encoding="utf-8"):
            r = json.loads(line)
            LOGBOOK[r["call"]] = r
    except Exception:
        pass


def _save_log():
    try:
        LOGFILE.parent.mkdir(exist_ok=True)
        with open(LOGFILE, "w", encoding="utf-8") as f:
            for r in LOGBOOK.values():
                f.write(json.dumps(r) + "\n")
    except Exception:
        pass


def log_calls(calls, band, khz, snr):
    """Log heard calls as PENDING (0 pts). Points come only after the verifier
    confirms the call is a real ham — a decode artifact that matches the pattern
    but isn't a licensed call never scores."""
    new = []
    for c in calls:
        if c in LOGBOOK:
            LOGBOOK[c]["count"] += 1
            if band not in LOGBOOK[c]["bands"]:
                LOGBOOK[c]["bands"].append(band)
                if LOGBOOK[c].get("verified"):
                    LOGBOOK[c]["points"] += 3        # new band on a real call
        else:
            LOGBOOK[c] = {"call": c, "first": time.strftime("%Y-%m-%d %H:%M"),
                          "bands": [band], "khz": khz, "count": 1, "snr": snr,
                          "verified": None, "name": "", "qth": "", "points": 0,
                          "tries": 0}
            new.append(c)
    if new:
        _save_log()
    return new


def verify_pending():
    """Check pending calls against the ham DB; score only the real ones."""
    changed = False
    for c, r in list(LOGBOOK.items()):
        if r.get("verified") is None and r.get("tries", 0) < 4:
            res = hamdb.verify(c)
            r["tries"] = r.get("tries", 0) + 1
            if res["status"] == "VALID":
                prefix = re.match(r"[A-Z0-9]*[0-9]", c).group()[:2]
                rare = 0 if any(v.get("verified") and v["call"] != c
                                and v["call"].startswith(prefix) for v in LOGBOOK.values()) else 5
                r.update({"verified": True, "name": res["name"], "qth": res["qth"],
                          "points": 10 + rare + 3 * (len(r["bands"]) - 1)})
                changed = True
            elif res["status"] == "INVALID":
                r["verified"] = False                # decode artifact — never scores
                changed = True
    if changed:
        _save_log()
    return changed


_KHZ2BAND = {3560: "80m", 5357: "60m", 7025: "40m", 10120: "30m", 14025: "20m",
             18075: "17m", 21025: "15m", 24905: "12m", 28025: "10m"}


def band_advice():
    """Data-driven band advisor: from our OWN harvest history (what this antenna
    actually heard, by band and UTC hour), tell the user where the CW is right now.
    Beats a generic propagation chart — it's personalized to this rig/location."""
    import glob
    import json
    from collections import Counter
    now = time.gmtime().tm_hour
    hrs = {(now - 1) % 24, now, (now + 1) % 24}
    by_band, now_band = Counter(), Counter()
    total = 0
    for jf in glob.glob(str(HARVEST_DIR / "*.json")):
        try:
            d = json.loads(Path(jf).read_text())
        except Exception:
            continue
        khz = d.get("khz", 0)
        b = _KHZ2BAND.get(min(_KHZ2BAND, key=lambda k: abs(k - khz)), f"{khz:.0f}")
        by_band[b] += 1; total += 1
        ts = d.get("trip_utc", "")
        try:
            if int(ts[11:13]) in hrs:
                now_band[b] += 1
        except (ValueError, IndexError):
            pass
    return {"now_utc_hour": now, "total": total,
            "best_now": now_band.most_common(4), "all_time": by_band.most_common(6)}


def log_summary():
    v = [c for c in LOGBOOK.values() if c.get("verified")]
    pend = sum(1 for c in LOGBOOK.values() if c.get("verified") is None)
    # progression stats — the things hams chase (prefixes~DXCC, bands, states/WAS)
    prefixes, bands, states = set(), set(), set()
    for c in v:
        m = re.match(r"[A-Z0-9]*[0-9]", c["call"])
        if m:
            prefixes.add(m.group())
        bands.update(c.get("bands", []))
        st = re.search(r"\b([A-Z]{2})\b", c.get("qth", "") or "")
        if st:
            states.add(st.group(1))
    return {"score": sum(c["points"] for c in v), "count": len(v), "pending": pend,
            "stats": {"prefixes": len(prefixes), "bands": len(bands), "states": len(states)},
            "calls": sorted(v, key=lambda c: c["first"], reverse=True)[:30]}


class SDRWorker(threading.Thread):
    daemon = True

    def run(self):
        self.aud_phase = 0.0
        self.zi = None
        self.agc = 1.0
        self.cw_off = 0.0
        while True:
            if not STATE["running"]:
                time.sleep(0.4); continue
            try:
                self._session()
            except Exception as e:
                STATE["err"] = str(e)[:120]; time.sleep(2.0)

    def _audio(self, iq):
        """Continuous-phase BFO -> narrow LP -> decimate -> int16 CW audio."""
        n = np.arange(len(iq), dtype=np.float64)
        off = STATE["lock_off"] if STATE["chlock"] else cur_off_hz()
        mixf = BFO_HZ - off                    # bring cursor's carrier to BFO pitch
        nco = np.exp(1j * (2 * np.pi * mixf / FS * n + self.aud_phase)).astype(np.complex64)
        self.aud_phase = (self.aud_phase + 2 * np.pi * mixf / FS * len(iq)) % (2 * np.pi)
        xr = (iq * nco).real.astype(np.float32)
        if self.zi is None:
            self.zi = lfilter_zi(_lp, 1.0).astype(np.float32) * xr[0]
        y, self.zi = lfilter(_lp, 1.0, xr, zi=self.zi)
        a = y[::AUD_DEC]
        pk = float(np.abs(a).max())
        self.agc = max(self.agc * 0.995, pk, 1e-4)
        a16 = np.clip(a / self.agc * 7000.0, -32767, 32767).astype(np.int16)
        with _alock:
            AUDIO.extend(a16)

    def _session(self):
        if radio_lock and not radio_lock.acquire("hamtuna_panel", "panel", 80, wait_s=30):
            STATE["lock"] = "busy"; time.sleep(3); return
        STATE["lock"] = "held"; STATE["err"] = ""
        from SoapySDR import SOAPY_SDR_RX
        sdr, st = cw._open_sdr(STATE["antenna"], FS)
        try:
            sdr.setGain(SOAPY_SDR_RX, 0, "IFGR", float(STATE["ifgr"]))
            sdr.writeSetting("rfgain_sel", str(STATE["rfsel"]))
        except Exception:
            pass
        cur = None
        buf = np.empty(2 * 65536, np.int16)
        last_off = 0.0
        while STATE["running"]:
            if radio_lock and radio_lock.should_yield():
                break
            if STATE["center_khz"] != cur:
                cur = STATE["center_khz"]
                sdr.setFrequency(SOAPY_SDR_RX, 0, cur * 1e3)
                ring_clear(); self.zi = None; time.sleep(0.15)
            r = sdr.readStream(st, [buf], 16384, timeoutUs=500000)  # small reads = fast waterfall (~15 rows/s)
            if r.ret <= 0:
                continue
            iq = ((buf[0:2 * r.ret:2].astype(np.float32)
                   + 1j * buf[1:2 * r.ret:2].astype(np.float32)) / 32768.0).astype(np.complex64)
            ring_write(iq)                          # fast; decode happens off-thread
            hi = _spectrum(iq)
            if hi is not None:
                ov = _downsample(hi, DISP_BINS)         # zoomed-out overview
                with _lock:
                    SPEC["hi"] = hi; SPEC["db"] = ov.tolist()
                    SPEC["peak_db"] = float(hi.max())
                    SPEC["noise_db"] = float(np.percentile(hi, 25)); SPEC["ts"] = time.time()
            if STATE["mode"] == "CW":
                self._audio(iq)            # BFO follows the cursor (cur_off_hz)
            if radio_lock:
                radio_lock.heartbeat()
        try:
            sdr.deactivateStream(st); sdr.closeStream(st); del sdr
        except Exception:
            pass
        if radio_lock:
            radio_lock.release("hamtuna_panel")
        STATE["lock"] = "released"


class Decoder(threading.Thread):
    """Off the read thread: every DECODE_EVERY s, snapshot a long COHERENT
    window from the ring and decode it as one piece (like the old cw.py listen),
    so transmissions aren't chopped into fragments and the reader never stalls."""
    daemon = True

    def run(self):
        while True:
            time.sleep(DECODE_EVERY)
            if not STATE["running"] or STATE["mode"] not in DECODERS:
                continue
            iqd = ring_snapshot(DECODE_SECS)
            if iqd is None:
                continue
            try:
                res = DECODERS[STATE["mode"]](iqd)
            except Exception as e:
                res = {"text": "", "wpm": 0, "q": 0, "conf": 0, "elements": 0,
                       "hint": f"decode err: {e}"[:80], "offset_hz": 0}
            res["mode"] = STATE["mode"]; res["ts"] = time.time()
            try:                                       # never let a post-decode bug kill this thread
                snr = 0.0
                with _lock:
                    DECODE.update(res)
                    if res["text"]:
                        TRANSCRIPT.append({"ts": time.strftime("%H:%M:%S"),
                                           "text": res["text"], "q": res["q"]})
                        snr = round(max(0, SPEC["peak_db"] - SPEC["noise_db"]), 1)
                if res.get("text") and res["mode"] == "CW":
                    got = log_calls(extract_calls(res["text"]), STATE["band"],
                                    STATE["center_khz"], snr)
                    if got:
                        res["new_calls"] = got         # log_calls returns a list of call strings
                        with _lock:
                            DECODE["new_calls"] = got
            except Exception as e:
                STATE["err"] = f"decode-post: {e}"[:100]


class Verifier(threading.Thread):
    """Off-thread: check pending logged calls against the ham DB (network),
    so verification never blocks decode."""
    daemon = True

    def run(self):
        while True:
            time.sleep(5)
            try:
                verify_pending()
            except Exception:
                pass


class Classifier(threading.Thread):
    """Off-thread: probe each detected carrier and tag whether it's actually
    copyable CW (valid WPM + real text) vs data/QSB/machine-CW — so the signal
    list tells you what you can READ, not just what's loud."""
    daemon = True

    def run(self):
        while True:
            time.sleep(8)
            if STATE["mode"] != "CW":
                continue
            iq = ring_snapshot(8)
            if iq is None:
                continue
            band = STATE["band"]
            out = []
            for s in detect_signals():
                co = (s["khz"] - STATE["center_khz"]) * 1000.0
                cw_ok, wpm, eye = False, 0, 0.0
                try:
                    ss = iq[:int(FS)]
                    n = np.arange(len(ss))
                    x = (ss * np.exp(-2j * np.pi * co / FS * n)).astype(np.complex64)
                    off = co + cw.find_offset(x, FS, 400)
                    env, aud = envelope_locked(iq, off)
                    txt, info = cw.decode_env_auto(env, aud)
                    w = info.get("wpm", 0)
                    eye = cw_quality.eye_opening(env)[0]
                    # copyable = a real CW eye is OPEN (Q>=readable) AND the keying
                    # rate is sane. Eye-based tag catches copyable CW even when the
                    # text decode is partial, and rejects data carriers (no CW eye).
                    cw_ok = bool(eye >= cw_quality.Q_READABLE and 3 <= w <= 45
                                 and len([c for c in txt if c != " "]) >= 3)
                    wpm = round(float(w), 1) if cw_ok else 0
                except Exception:
                    pass
                out.append({"khz": s["khz"], "snr": s["snr"], "cw": cw_ok,
                            "wpm": wpm, "eye": round(float(eye), 1)})
            # the classifier is the signal-list authority: detect + decode in ONE
            # pass, so tags always match their carrier (no cross-snapshot mismatch)
            with _lock:
                SIGLIST["band"] = band; SIGLIST["sigs"] = out


def _wav_header(nbytes=0x7FFFF000):
    return (b"RIFF" + struct.pack("<I", nbytes + 36) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, AUD_FS, AUD_FS * 2, 2, 16) +
            b"data" + struct.pack("<I", nbytes))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/":
            self._send(PAGE, "text/html; charset=utf-8")
        elif u.path == "/spectrum":
            # optional zoom view: vc/vs = view center/span kHz. Grab refs under the
            # lock, then do the numpy slicing OUTSIDE it (hi is replaced whole each
            # reader cycle, never mutated in place) so the draw loop can't contend
            # with the SDR reader and freeze the server.
            with _lock:
                center = STATE["center_khz"]; hi = SPEC["hi"]; ov = SPEC["db"]
                peak, noise = SPEC["peak_db"], SPEC["noise_db"]
            try:
                vs = float(q["vs"][0]); vc = float(q["vc"][0])
            except (KeyError, ValueError):
                vs = vc = None
            if vs is not None and hi is not None and vs < SPAN_KHZ - 1:
                sl, actc, acts = _view_slice(hi, center, vc, vs)
                out = {"db": sl.tolist(), "peak": float(sl.max()),
                       "noise": float(np.percentile(sl, 25)), "center": actc, "span": acts}
            else:
                out = {"db": ov, "peak": peak, "noise": noise,
                       "center": center, "span": SPAN_KHZ}
            self._send(json.dumps(out))
        elif u.path == "/state":
            with _lock:
                self._send(json.dumps({**{k: STATE[k] for k in
                    ("center_khz", "tune_khz", "band", "mode", "ifgr", "running",
                     "lock", "err", "chlock")}, "span": SPAN_KHZ,
                    "decode": dict(DECODE), "bands": BANDS, "modes": MODES,
                    "transcript": list(TRANSCRIPT)[-14:],
                    "smeter": round(max(0, (SPEC["peak_db"] - SPEC["noise_db"])), 1)}))
        elif u.path == "/set":
            if "band" in q and q["band"][0] in BANDS:
                STATE["band"] = q["band"][0]
                STATE["center_khz"] = float(BANDS[q["band"][0]])
                STATE["tune_khz"] = STATE["center_khz"]     # cursor to band center
                STATE["chlock"] = False
            if "center" in q:
                try:
                    STATE["center_khz"] = round(float(q["center"][0]), 2)
                    STATE["tune_khz"] = STATE["center_khz"]; STATE["chlock"] = False
                except ValueError: pass
            if "mode" in q and q["mode"][0] in MODES:
                STATE["mode"] = q["mode"][0]
            if "running" in q:
                STATE["running"] = q["running"][0] == "1"
            self._send(json.dumps({"ok": True}))
        elif u.path == "/autotune":
            # jump the cursor to the best COPYABLE CW (highest eye-opening), not the
            # loudest carrier (which is usually FT8/data). Falls back to strongest
            # only if the classifier hasn't found any open-eye CW yet.
            with _lock:
                fresh = SIGLIST["band"] == STATE["band"] and SIGLIST["sigs"]
                sigs = [dict(s) for s in SIGLIST["sigs"]] if fresh else detect_signals()
            cw_sigs = [s for s in sigs if s.get("cw")]
            pick = (max(cw_sigs, key=lambda s: s.get("eye", 0)) if cw_sigs
                    else (max(sigs, key=lambda s: s["snr"]) if sigs else None))
            if pick:
                STATE["tune_khz"] = pick["khz"]; STATE["chlock"] = False
            self._send(json.dumps({"ok": True, "tune": STATE["tune_khz"],
                                   "found_cw": bool(cw_sigs), "n_cw": len(cw_sigs)}))
        elif u.path == "/tune":                # move the CURSOR within the window
            try:
                khz = float(q["khz"][0])
                # snap=1: refine to the TRUE peak in the high-res (~30 Hz) spectrum
                # near the click, so the cursor lands exactly on the signal and stays
                # locked to it at any zoom (coarse display bins otherwise drift on zoom)
                if q.get("snap", ["0"])[0] == "1":
                    with _lock:
                        hi = SPEC["hi"]; c = STATE["center_khz"]
                    if hi is not None and len(hi):
                        binkhz = SPAN_KHZ / len(hi)
                        lo0 = c - SPAN_KHZ / 2
                        i = int(round((khz - lo0) / binkhz))
                        # search window scales with zoom (passed as 'win' kHz): tiny
                        # when zoomed in so the cursor lands where you clicked, only
                        # nudging onto the exact peak.
                        try:
                            winkhz = min(3.0, max(0.05, float(q["win"][0])))
                        except (KeyError, ValueError):
                            winkhz = 0.15
                        win = max(1, int(winkhz / binkhz))
                        a = max(0, i - win); b = min(len(hi), i + win + 1)
                        if b > a:
                            bi = a + int(np.argmax(hi[a:b]))
                            khz = lo0 + bi * binkhz
                lo = STATE["center_khz"] - SPAN_KHZ / 2 + 0.5
                hi_k = STATE["center_khz"] + SPAN_KHZ / 2 - 0.5
                STATE["tune_khz"] = round(min(hi_k, max(lo, khz)), 3)
                STATE["chlock"] = False
            except (ValueError, KeyError): pass
            self._send(json.dumps({"ok": True, "tune": STATE["tune_khz"]}))
        elif u.path == "/signals":
            with _lock:
                fresh = SIGLIST["band"] == STATE["band"] and SIGLIST["sigs"]
                sigs = [dict(s) for s in SIGLIST["sigs"]] if fresh else None
            if sigs is None:            # band just changed: show carriers now, tags fill in <=8s
                sigs = detect_signals()
                for s in sigs:
                    s["cw"] = None; s["wpm"] = 0
            self._send(json.dumps({"signals": sigs,
                                   "center": STATE["center_khz"], "tune": STATE["tune_khz"]}))
        elif u.path == "/log":
            self._send(json.dumps(log_summary()))
        elif u.path == "/advisor":
            self._send(json.dumps(band_advice()))
        elif u.path == "/lock":
            if q.get("on", ["1"])[0] == "1":
                STATE["lock_off"] = cur_off_hz()      # pin the cursor; decode snaps ±400 to its carrier
                STATE["chlock"] = True
            else:
                STATE["chlock"] = False
            self._send(json.dumps({"ok": True, "chlock": STATE["chlock"]}))
        elif u.path == "/step":                # step the cursor to prev/next signal
            d = q.get("d", ["1"])[0]
            freqs = sorted(s["khz"] for s in detect_signals())
            if freqs:
                t = STATE["tune_khz"]
                STATE["tune_khz"] = (next((f for f in freqs if f > t + 0.25), freqs[0]) if d == "1"
                                     else next((f for f in reversed(freqs) if f < t - 0.25), freqs[-1]))
                STATE["chlock"] = False
            self._send(json.dumps({"ok": True, "tune": STATE["tune_khz"]}))
        elif u.path == "/cw_audio.wav":
            self._stream_audio()
        else:
            self.send_response(404); self.end_headers()

    def _stream_audio(self):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(_wav_header())
            while STATE["running"]:
                with _alock:
                    chunk = bytes(np.array(AUDIO, np.int16).tobytes()) if AUDIO else b""
                    AUDIO.clear()
                if chunk:
                    self.wfile.write(chunk)
                else:
                    self.wfile.write(b"\x00\x00" * (AUD_FS // 20))   # 50 ms silence keeps it flowing
                time.sleep(0.05)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8647)
    args = ap.parse_args()
    os.environ["PATH"] = r"C:\Program Files\SDRplay\API\x64" + os.pathsep + os.environ.get("PATH", "")
    _load_log()
    SDRWorker().start()
    Decoder().start()
    Verifier().start()
    Classifier().start()
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>hamTuna</title><style>
:root{--bg:#000;--panel:#080c11;--ink:#d6e6f2;--mut:#5f7893;--acc:#2ee6c8;--acc2:#ff5d73;--hair:#141f2b;--good:#3ad17a;--warn:#f0b23a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:ui-monospace,Consolas,monospace;overflow:hidden}
.top{display:flex;align-items:center;gap:14px;padding:8px 14px;border-bottom:1px solid var(--hair);background:var(--panel)}
.logo{font-weight:700;letter-spacing:.06em;color:var(--acc);font-size:18px}.logo b{color:var(--acc2)}
.freq{font-size:30px;font-weight:700;letter-spacing:.04em;color:#fff;text-shadow:0 0 14px rgba(46,230,200,.4)}
.freq small{font-size:13px;color:var(--mut)}.sub{color:var(--mut);font-size:12px}
.wrap{display:grid;grid-template-columns:1fr 320px;height:calc(100vh - 52px)}
.left{display:flex;flex-direction:column;min-width:0;position:relative}
.curline{position:absolute;top:0;bottom:0;width:2px;pointer-events:none;background:#fff;color:#ffd84a;box-shadow:0 0 6px currentColor;z-index:5}
.curline::before{content:'';position:absolute;top:0;left:-5px;border-left:6px solid transparent;border-right:6px solid transparent;border-top:9px solid currentColor}
.curline::after{content:'';position:absolute;bottom:0;left:-5px;border-left:6px solid transparent;border-right:6px solid transparent;border-bottom:9px solid currentColor}
.zoomctl{position:absolute;top:8px;right:10px;display:flex;flex-direction:column;gap:5px;z-index:6}
.zoomctl button{width:32px;height:32px;font-size:17px;font-weight:700;background:rgba(8,22,28,.88);color:#7fd6e6;border:1px solid #1c4a58;border-radius:6px;cursor:pointer;line-height:1}
.zoomctl button:hover{background:#14303a;color:#aeeaf5}
#spec{background:#000;flex:0 0 190px;width:100%;cursor:crosshair}#wf{background:#000;flex:1;width:100%;cursor:crosshair}
.side{border-left:1px solid var(--hair);background:var(--panel);padding:12px;overflow-y:auto;display:flex;flex-direction:column;gap:13px}
.row{display:flex;flex-wrap:wrap;gap:6px}
button{background:#0c161f;color:var(--ink);border:1px solid var(--hair);border-radius:7px;padding:7px 10px;font-family:inherit;font-size:12px;cursor:pointer}
button:hover{border-color:var(--acc)}button.on{background:var(--acc);color:#04110e;border-color:var(--acc);font-weight:700}
button.mode.on{background:var(--acc2);color:#1a0409;border-color:var(--acc2)}
.lbl{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);margin-bottom:5px}
.dial{background:#060d13;border:1px solid var(--hair);border-radius:10px;padding:12px}
.gauge{height:10px;background:#08120f;border-radius:6px;overflow:hidden;margin:6px 0}
.gfill{height:100%;background:linear-gradient(90deg,var(--acc2),var(--warn),var(--good));transition:width .3s}
.big{font-size:22px;font-weight:700}
.xscript{background:#000;border:1px solid var(--hair);border-radius:8px;padding:10px;height:150px;overflow-y:auto;font-size:15px;line-height:1.7;letter-spacing:.04em}
.xline{color:var(--acc);word-break:break-word}
.stat{display:flex;justify-content:space-between;font-size:12px;color:var(--mut);padding:2px 0}.stat b{color:var(--ink)}
.autob{background:var(--acc);color:#04110e;font-weight:700;width:100%;padding:10px;font-size:13px}
.advb{background:#12303a;color:#7fd6e6;border:1px solid #1c4a58;width:100%;padding:8px;font-size:12px;margin-top:2px}
.advout{font-size:12px}.advrow{display:flex;gap:6px;flex-wrap:wrap;margin:3px 0}
.advband{background:#0b141c;border:1px solid #14303a;padding:3px 8px;border-radius:10px;cursor:pointer}
.advband:hover{background:#14303a}.advband b{color:var(--good)}
.listen{width:100%;padding:9px;font-size:13px;font-weight:700}.listen.on{background:var(--acc2);color:#1a0409;border-color:var(--acc2)}
.lockb{width:100%;padding:9px;font-size:13px;font-weight:700}.lockb.on{background:var(--warn);color:#1a1204;border-color:var(--warn)}
.siglist{background:#000;border:1px solid var(--hair);border-radius:8px;max-height:150px;overflow-y:auto}
.sig{display:flex;align-items:center;gap:8px;padding:5px 9px;font-size:12px;cursor:pointer;border-bottom:1px solid #0b141c}
.sig:last-child{border-bottom:none}.sig:hover{background:#0b141c}.sig.on{background:rgba(46,230,200,.12);color:var(--acc)}
.sig .bar{flex:1;height:5px;background:#08120f;border-radius:3px;overflow:hidden}.sig .bar span{display:block;height:100%;background:var(--acc)}
.sig .snr{color:var(--mut);font-size:10px;width:40px;text-align:right}
.sig.dim{opacity:.5}.cwtag{font-size:9px;color:var(--good);font-weight:700;white-space:nowrap}.cwtag.off{color:var(--mut);font-weight:400}
button.step{padding:2px 9px;font-size:13px;font-weight:700}
.logbook{background:#060d13;border:1px solid var(--hair);border-radius:10px;padding:12px}
.score{font-size:28px;font-weight:700;color:var(--good);text-shadow:0 0 12px rgba(58,209,122,.3)}.score small{font-size:12px;color:var(--mut);margin-left:5px}
.logstats{display:flex;gap:10px;flex-wrap:wrap;font-size:11px;color:var(--mut);margin:4px 0 2px}
.logstats span{background:#0b141c;padding:2px 7px;border-radius:10px;cursor:default}
.loglist{max-height:150px;overflow-y:auto;margin-top:6px}
.logrow{display:flex;justify-content:space-between;gap:8px;font-size:12px;padding:4px 0;border-bottom:1px solid #0b141c}
.logrow:last-child{border-bottom:none}.logrow .call{color:var(--acc);font-weight:700}.logrow .meta{color:var(--mut);font-size:10px}
.newcall{color:var(--good);font-weight:700;font-size:13px;min-height:16px}
.smeter{height:8px;background:#08120f;border-radius:5px;overflow:hidden}.sfill{height:100%;background:var(--acc);transition:width .2s}
.chip{font-size:10px;padding:2px 7px;border-radius:20px;border:1px solid var(--hair);color:var(--mut)}
.chip.held{color:var(--good);border-color:var(--good)}.chip.busy{color:var(--warn);border-color:var(--warn)}
</style></head><body>
<div class=top>
  <div class=logo>ham<b>Tuna</b></div>
  <div class=freq id=freq>14030.00<small> kHz</small></div>
  <div class=sub id=bandlbl>20m &middot; CW</div><div style=flex:1></div>
  <span class="chip" id=lock>lock</span>
  <div class=sub>click a signal &rarr; snap to its peak</div>
</div>
<div class=wrap>
  <div class=left><canvas id=spec></canvas><canvas id=wf></canvas><div class=curline id=curline></div>
    <div class=zoomctl>
      <button onclick="zoomBy(0.6,0.5)" title="zoom in (narrower band)">+</button>
      <button onclick="zoomBy(1.7,0.5)" title="zoom out">&minus;</button>
      <button onclick="centerCursor()" title="center on tuning cursor">&#9678;</button>
      <button onclick="viewReset()" title="full band (double-click waterfall)">&#10530;</button>
    </div></div>
  <div class=side>
    <div><div class=lbl>Band</div><div class=row id=bands></div></div>
    <button class=advb onclick=advise()>📡 Where's the CW right now?</button>
    <div class=advout id=advout></div>
    <div><div class=lbl>Mode</div><div class=row id=modes></div></div>
    <button class=autob id=autob onclick=autotune()>&#9673; AUTO-TUNE (best copyable CW)</button>
    <button class=lockb id=lockb onclick=togLock()>&#128275; LOCK channel</button>
    <div>
      <div class=lbl style="display:flex;justify-content:space-between;align-items:center">
        <span>CW signals on air</span>
        <span><button class=step onclick="step(-1)">&#9664;</button> <button class=step onclick="step(1)">&#9654;</button></span>
      </div>
      <div class=siglist id=siglist></div>
    </div>
    <button class=listen id=listenb onclick=togListen()>&#9654; LISTEN (live audio)</button>
    <audio id=au></audio>
    <div class=dial>
      <div class=lbl>Copy Quality &mdash; eye-opening (the CW "MER")</div>
      <div class=big><span id=verdict>&mdash;</span> <small id=copypct class=sub></small></div>
      <div class=gauge><div class=gfill id=eyebar style=width:0%></div></div>
      <div class=stat><span>eye-opening Q</span><b id=eye>&mdash;</b></div>
      <div class=stat><span>WPM</span><b id=wpm>&mdash;</b></div>
      <div class=stat><span>decoder</span><b id=route>&mdash;</b></div>
      <div class=stat><span>decode confidence</span><b id=conf>0%</b></div>
      <div class=stat><span>S-meter</span><b id=sm>&mdash;</b></div>
      <div class=smeter><div class=sfill id=smbar style=width:0%></div></div>
    </div>
    <div><div class=lbl id=declbl>Live Morse transcript</div><div class=xscript id=xscript></div></div>
    <div class=newcall id=newcall></div>
    <div class=logbook>
      <div class=lbl style="display:flex;justify-content:space-between;align-items:baseline">
        <span>&#128225; Logbook</span><span id=logcount class=sub></span></div>
      <div class=score><span id=score>0</span><small>pts</small></div>
      <div class=logstats id=logstats></div>
      <div class=loglist id=loglist></div>
    </div>
    <div class=sub id=hint></div>
  </div>
</div>
<script>
let ST={};const $=id=>document.getElementById(id);
const spec=$('spec'),sx=spec.getContext('2d'),wf=$('wf'),wx=wf.getContext('2d');
function fit(){for(const c of [spec,wf]){c.width=c.clientWidth;c.height=c.clientHeight;}}
addEventListener('resize',fit);fit();
async function api(p){return (await fetch(p)).json();}
let DB=[];
// waterfall navigation: VIEW = the visible freq window (c=center kHz, s=span kHz).
// null span = full band. VC/VS = the actual window the last /spectrum returned.
let VIEW={c:null,s:null}, VC=null, VS=null, FULLSPAN=250, SDRCENTER=null, wfDirty=false;
let prevVC=null, prevVS=null;   // for real-time horizontal waterfall pan
function viewInit(center,span){FULLSPAN=span;SDRCENTER=center;
  if(VIEW.c===null){VIEW.c=center;VIEW.s=span;}}
function viewReset(){if(SDRCENTER!==null){VIEW.c=SDRCENTER;VIEW.s=FULLSPAN;wfDirty=true;}}
function centerCursor(){if(ST.tune_khz){VIEW.c=ST.tune_khz;clampView();wfDirty=true;}}  // jump back to the cursor
function clampView(){
  VIEW.s=Math.max(1,Math.min(FULLSPAN,VIEW.s));   // clamp span FIRST (1 kHz deepest zoom)
  if(VIEW.s>=FULLSPAN*0.985){VIEW.s=FULLSPAN;VIEW.c=SDRCENTER;return;}  // snap to full band so zoom-out always reaches it
  const half=VIEW.s/2, lo=SDRCENTER-FULLSPAN/2+half, hi=SDRCENTER+FULLSPAN/2-half;
  VIEW.c=Math.max(lo,Math.min(hi,VIEW.c));}
function zoomBy(factor,fx){          // fx = 0..1 anchor point across the canvas
  if(SDRCENTER===null||VC===null)return;
  const fk=VC-VS/2+fx*VS;            // freq under the anchor right now
  VIEW.s=VS*factor;clampView();
  VIEW.c=fk-(fx-0.5)*VIEW.s;clampView();   // hold that freq under the anchor
  wfDirty=true;}                     // rebuild the waterfall at the new scale
function onWheel(e){e.preventDefault();
  // Zoom PIVOTS on the tuning cursor: the locked signal stays exactly where it is on
  // screen and the band expands/contracts around it. The cursor only ever moves when
  // you click a signal, never while zooming.
  if(VC===null||VS===null)return;
  const tk=(ST.tune_khz!=null?ST.tune_khz:VC);
  const fx=(tk-(VC-VS/2))/VS;                    // cursor's current position across the view
  VIEW.s=Math.max(1,Math.min(FULLSPAN,VS*(e.deltaY>0?1.5:0.66)));
  VIEW.c=tk-(fx-0.5)*VIEW.s;                     // hold the cursor at that same spot
  clampView();wfDirty=true;}                     // scroll UP = zoom IN
// middle-drag pan: slide the view LIVE (VIEW.c updates every move; the waterfall
// history shifts horizontally in draw() so it slides in real time). mousemove/up on
// WINDOW so it tracks even when the mouse leaves the canvas.
let panning=false,panStartX=0,panStartC=0;
function onPanStart(e){if(e.button!==1)return;e.preventDefault();
  panning=true;panStartX=e.clientX;panStartC=VIEW.c;document.body.style.userSelect='none';}
function onPanMove(e){if(!panning||!VS||!spec.width)return;
  const dkhz=(e.clientX-panStartX)/spec.width*VS;   // drag right -> lower freqs (grab & pull)
  VIEW.c=panStartC-dkhz;clampView();}
function onPanEnd(){if(!panning)return;panning=false;document.body.style.userSelect='';}
let lastCenter=null;
async function refresh(){
  ST=await api('/state');
  if(lastCenter!==null&&Math.abs(ST.center_khz-lastCenter)>0.01){   // band changed -> reset zoom
    SDRCENTER=ST.center_khz;FULLSPAN=ST.span;viewReset();}
  lastCenter=ST.center_khz;
  $('freq').innerHTML=(ST.tune_khz||ST.center_khz).toFixed(2)+'<small> kHz</small>';
  $('bandlbl').textContent=ST.band+' · '+ST.mode;
  const lk=$('lock');lk.textContent=ST.lock;lk.className='chip '+ST.lock;
  $('lockb').classList.toggle('on',ST.chlock);
  $('lockb').innerHTML=ST.chlock?'&#128274; LOCKED — following QSO':'&#128275; LOCK channel';
  $('freq').style.color=ST.chlock?'#f0b23a':'#fff';
  if(!$('bands').dataset.f){$('bands').dataset.f=1;
    for(const b in ST.bands){const e=document.createElement('button');e.textContent=b;e.onclick=()=>set('band='+b);e.dataset.b=b;$('bands').appendChild(e);}
    ST.modes.forEach(m=>{const e=document.createElement('button');e.className='mode';e.textContent=m;e.onclick=()=>set('mode='+m);e.dataset.m=m;$('modes').appendChild(e);});}
  [...$('bands').children].forEach(e=>e.classList.toggle('on',e.dataset.b===ST.band));
  [...$('modes').children].forEach(e=>e.classList.toggle('on',e.dataset.m===ST.mode));
  const d=ST.decode||{};
  const vd=d.verdict||'—', vc={SOLID:'#33ff99',READABLE:'#f0d24a',FAILING:'#ff5a5a'}[vd]||'#8aa';
  const vel=$('verdict'); vel.textContent=vd; vel.style.color=vc;
  $('copypct').textContent=(d.copy_pct!=null&&d.wpm)?('~'+d.copy_pct+'% copy'):'';
  const eq=d.eye_q||0; $('eye').textContent=eq?eq.toFixed(2):'—';
  $('eyebar').style.width=Math.min(100,Math.max(0,(eq-1.5)/(4.0-1.5)*100))+'%';
  $('eyebar').style.background=vc;
  $('conf').textContent=Math.round((d.conf||0)*100)+'%';
  $('route').textContent=d.route?({neural:'🧠 neural AI',mf:'matched-filter (fading)',classic:'classic'}[d.route]||d.route):'—';
  $('wpm').textContent=d.wpm?d.wpm.toFixed(1):'—';
  $('sm').textContent=(ST.smeter||0).toFixed(0)+' dB';$('smbar').style.width=Math.min(100,(ST.smeter||0)*2.2)+'%';
  $('declbl').textContent=ST.mode==='CW'?'Live Morse transcript':ST.mode+' decode';
  const xs=$('xscript');
  if(ST.mode==='CW'){
    xs.innerHTML = d.text ? `<div class=xline>${d.text}</div>` : '<div class=sub>…listening for CW…</div>';
    xs.scrollTop=xs.scrollHeight;
  } else xs.innerHTML='<div class=sub>'+ST.mode+' decode coming soon — spectrum + audio live</div>';
  $('hint').textContent=d.hint||'';
  $('newcall').textContent=(d.new_calls&&d.new_calls.length)?('🎉 logged '+d.new_calls.join(' ')):'';
}
async function pollLog(){let s;try{s=await api('/log');}catch(e){return;}
  $('score').textContent=s.score;
  $('logcount').textContent=s.count+' verified'+(s.pending?(' · '+s.pending+' pending'):'');
  const st=s.stats||{};$('logstats').innerHTML=
    `<span title="unique callsign prefixes (~DXCC)">🌐 ${st.prefixes||0} pfx</span>`+
    `<span title="bands worked">📶 ${st.bands||0} bands</span>`+
    `<span title="US states worked (WAS)">🗺️ ${st.states||0} states</span>`;
  $('loglist').innerHTML=(s.calls||[]).length?(s.calls).map(c=>
    `<div class=logrow><span class=call>&check; ${c.call}</span><span class=meta>${(c.name||'').split(' ')[0]} &middot; ${c.bands.join('/')} &middot; ${c.points}pt</span></div>`).join('')
    :'<div class=sub>no verified calls yet — tune in a CQ</div>';}
async function set(kv){await api('/set?'+kv);refresh();}
async function autotune(){
  const b=$('autob'), lbl=b.textContent;
  b.textContent='… searching for copyable CW …';
  let r; try{r=await api('/autotune');}catch(e){b.textContent=lbl;return;}
  await refresh();
  // real click feedback: report what it found (no silent no-op)
  b.textContent = r.found_cw ? ('◉ jumped to CW ('+r.n_cw+' copyable)') : '◉ no copyable CW here — try 40m / evening';
  setTimeout(()=>{b.textContent=lbl;}, 2200);
}
async function advise(){let s;try{s=await api('/advisor');}catch(e){return;}
  const o=$('advout');
  const fmt=a=>a.length?a.map(([b,n])=>`<span class=advband onclick="set('band='+'${b}')">${b} <b>${n}</b></span>`).join(''):'<span class=sub>no history yet</span>';
  o.innerHTML=`<div class=sub>Your rig's CW, this hour (${String(s.now_utc_hour).padStart(2,'0')}:00Z), from ${s.total} captures:</div>`+
    `<div class=advrow>${fmt(s.best_now)}</div>`+
    `<div class=sub style="margin-top:4px">all-time by band:</div><div class=advrow>${fmt(s.all_time)}</div>`+
    (s.best_now.length?'<div class=sub style="margin-top:4px">click a band to jump there</div>':'<div class=sub style="margin-top:4px">quiet this hour — CW peaks evenings/weekends on 40/80m</div>');}
async function step(d){await api('/step?d='+(d>0?1:0));refresh();}
async function togLock(){await api('/lock?on='+(ST.chlock?0:1));refresh();}
async function tune(khz){await api('/tune?khz='+khz);refresh();}
async function pollSignals(){let s;try{s=await api('/signals');}catch(e){return;}
  const list=$('siglist'),sigs=s.signals||[],c=s.center;
  const cur=s.tune!==undefined?s.tune:c;
  list.innerHTML=sigs.length?sigs.map(x=>{const on=Math.abs(x.khz-cur)<0.3;
    const tag=x.cw===true?`<span class=cwtag>&check;CW ${x.wpm}</span>`:(x.cw===false?'<span class="cwtag off">data/busy</span>':'<span class="cwtag off">…</span>');
    return `<div class="sig${on?' on':''}${x.cw===false?' dim':''}" onclick="tune(${x.khz})"><span>${x.khz.toFixed(2)}</span>${tag}<span class=bar><span style="width:${Math.min(100,x.snr*3)}%"></span></span><span class=snr>${x.snr}dB</span></div>`;}).join(''):'<div class=sub style="padding:8px">no CW carriers here — try another band</div>';}
// left-click -> lock the cursor onto the clicked signal. Send the exact clicked
// freq and let the backend snap to the TRUE high-res peak (~30 Hz), so the cursor
// sits precisely on the signal and stays locked to it through any zoom.
function snap(e,c){if(e.button&&e.button!==0)return;if(VC===null)return;
  const r=c.getBoundingClientRect();const fx=(e.clientX-r.left)/r.width;
  const f=VC-VS/2+fx*VS;                  // the mouse IS the target: cursor goes exactly here
  api('/tune?khz='+f.toFixed(3)).then(refresh);}  // (decoder finds the carrier within +/-700Hz on its own)
// LEFT click/drag = grab & drag the tuning line exactly under the pointer.
let curDrag=false, dragKhz=null, lastSend=0;
function clientKhz(clientX){const r=spec.getBoundingClientRect();
  const fx=Math.max(0,Math.min(1,(clientX-r.left)/r.width));return VC-VS/2+fx*VS;}
function onCurDown(e){if(e.button!==0||VC===null)return;e.preventDefault();curDrag=true;onCurMove(e);}
function onCurMove(e){if(!curDrag)return;dragKhz=clientKhz(e.clientX);
  const now=performance.now();                       // throttle the tune requests during drag
  if(now-lastSend>70){lastSend=now;api('/tune?khz='+dragKhz.toFixed(3));}}
function onCurUp(){if(!curDrag)return;curDrag=false;
  if(dragKhz!=null){tune(dragKhz.toFixed(3));dragKhz=null;}}
// Controls: SCROLL = zoom, LEFT click/drag = tune (grab the line), MIDDLE-DRAG = pan,
// DOUBLE-CLICK = reset. (Middle-click autoscroll suppressed.)
for(const c of [spec,wf]){
  c.addEventListener('wheel',onWheel,{passive:false});
  c.addEventListener('mousedown',e=>{if(e.button===1){e.preventDefault();onPanStart(e);}else if(e.button===0)onCurDown(e);});
  c.addEventListener('dblclick',e=>{e.preventDefault();viewReset();});
  c.addEventListener('auxclick',e=>{if(e.button===1)e.preventDefault();});   // kill autoscroll
  c.addEventListener('contextmenu',e=>e.preventDefault());
}
addEventListener('mousemove',e=>{onPanMove(e);onCurMove(e);});   // WINDOW so drags track off-canvas
addEventListener('mouseup',e=>{onPanEnd(e);onCurUp(e);});
let listening=false;
function togListen(){const a=$('au');listening=!listening;$('listenb').classList.toggle('on',listening);
  if(listening){a.src='/cw_audio.wav?'+Date.now();a.play().catch(()=>{});$('listenb').innerHTML='&#9632; STOP audio';}
  else{a.pause();a.removeAttribute('src');a.load();$('listenb').innerHTML='&#9654; LISTEN (live audio)';}}
// OLED colormap: weak -> pure black, strong -> cyan -> white-hot
function oled(t){t=Math.max(0,Math.min(1,t));const g2=Math.pow(t,1.4);
  const r=255*Math.pow(Math.max(0,(g2-0.5)*2),1.3),g=255*Math.min(1,g2*1.85),b=255*Math.min(1,g2*1.6);
  return[r|0,g|0,b|0];}
async function draw(){
  const zoomed=(VIEW.c!==null&&VIEW.s<FULLSPAN-0.5);
  const q=zoomed?('?vc='+VIEW.c.toFixed(2)+'&vs='+VIEW.s.toFixed(2)):'';
  let s;try{s=await api('/spectrum'+q);}catch(e){setTimeout(draw,300);return;}
  if(!q){SDRCENTER=s.center;FULLSPAN=s.span;if(VIEW.c===null){VIEW.c=s.center;VIEW.s=s.span;}}
  const db=s.db;DB=db;VC=s.center;VS=s.span;if(!db.length){setTimeout(draw,150);return;}
  const w=spec.width,h=spec.height;sx.clearRect(0,0,w,h);
  sx.strokeStyle='#0c1a22';for(let i=0;i<=4;i++){const y=h*i/4;sx.beginPath();sx.moveTo(0,y);sx.lineTo(w,y);sx.stroke();}
  const lo=s.noise-6,hi=s.peak+6,rng=Math.max(6,hi-lo);
  sx.strokeStyle='#2ee6c8';sx.lineWidth=1.4;sx.shadowColor='#2ee6c8';sx.shadowBlur=6;sx.beginPath();
  for(let i=0;i<db.length;i++){const x=i/db.length*w,y=h-(db[i]-lo)/rng*h;i?sx.lineTo(x,y):sx.moveTo(x,y);}sx.stroke();sx.shadowBlur=0;
  const cwd=wf.width,ch=wf.height;
  if(wfDirty){wx.fillStyle='#000';wx.fillRect(0,0,cwd,ch);wfDirty=false;prevVC=VC;prevVS=VS;}  // rebuild at new zoom
  // real-time PAN: shift the waterfall history horizontally to stay freq-aligned
  let shiftPx=0;
  if(prevVC!==null&&Math.abs(VS-prevVS)<1e-9)shiftPx=Math.round((prevVC-VC)/VS*cwd);
  prevVC=VC;prevVS=VS;
  wx.putImageData(wx.getImageData(0,0,cwd,ch),shiftPx,1);   // scroll down 1 + pan horizontally
  if(shiftPx>0){wx.fillStyle='#000';wx.fillRect(0,0,shiftPx,ch);}          // clear newly-exposed edge
  else if(shiftPx<0){wx.fillStyle='#000';wx.fillRect(cwd+shiftPx,0,-shiftPx,ch);}
  const row=wx.createImageData(cwd,1);
  for(let x=0;x<cwd;x++){const i=Math.floor(x/cwd*db.length);let v=(db[i]-lo)/rng;const c=oled(v);
    row.data[x*4]=c[0];row.data[x*4+1]=c[1];row.data[x*4+2]=c[2];row.data[x*4+3]=255;}
  wx.putImageData(row,0,0);
  // tuning cursor overlay (spans spectrum + waterfall); hidden if panned off-view.
  // A top marker triangle makes the locked freq findable even when the signal is
  // sub-pixel-thin behind the line (e.g. zoomed all the way out).
  const cl=$('curline');
  const tk=(curDrag&&dragKhz!=null)?dragKhz:ST.tune_khz;   // follow the pointer live while dragging
  if(tk&&VS){const cx=(tk-(VC-VS/2))/VS*w;
    const vis=cx>=-1&&cx<=w+1;cl.style.display=vis?'block':'none';
    const col=ST.chlock?'#f0b23a':'#ffd84a';        // amber locked / yellow otherwise (high contrast on the OLED waterfall)
    cl.style.left=cx+'px';cl.style.background=col;cl.style.color=col;
    if(!vis){                                        // cursor panned off-view -> edge arrow pointing to it (click to recenter)
      sx.fillStyle=col;sx.font='bold 15px system-ui';sx.textAlign=cx<0?'left':'right';
      sx.fillText((cx<0?'◀ cursor':'cursor ▶'),cx<0?6:w-6,30);sx.textAlign='left';}}
  // frequency axis — the kHz labels visibly compress as you zoom (clear feedback)
  sx.fillStyle='rgba(150,185,205,.75)';sx.font='10px ui-monospace,monospace';sx.textAlign='center';
  for(let i=0;i<=4;i++){const fk=VC-VS/2+i/4*VS,x=Math.max(24,Math.min(w-24,i/4*w));
    sx.fillText(fk.toFixed(VS<40?2:0),x,h-19);}
  sx.textAlign='left';
  // zoom readout + controls hint (top-left of the spectrum)
  sx.fillStyle='rgba(120,200,220,.9)';sx.font='11px ui-monospace,monospace';
  const zt=(VS<FULLSPAN-0.5)?('🔍 zoom '+VS.toFixed(1)+' kHz span'):('full band '+VS.toFixed(0)+' kHz');
  sx.fillText(zt,8,15);
  sx.fillStyle='rgba(120,150,170,.6)';
  sx.fillText('scroll: zoom · ←/→: tune · ↑/↓: zoom · drag: pan · dblclick: reset',8,h-6);
  setTimeout(draw,60);   // fast waterfall / live feedback (~15 rows/s)
}
refresh();setInterval(refresh,1500);draw();
setInterval(pollSignals,2500);pollSignals();
setInterval(pollLog,3000);
</script></body></html>"""

if __name__ == "__main__":
    main()
