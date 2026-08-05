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
import cw_map
import cw_quality
import hamdb
try:
    import radio_lock
except Exception:
    radio_lock = None

FS = 250_000.0  # rate-ok: HF panel on cw.py's magnitude/envelope path, verified
#                 working daily; 250 kHz IS the panel's span contract (SPAN_KHZ)
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
         "lock": "none", "err": "", "chlock": False, "lock_off": 0.0, "last_off": 0.0,
         "cw_filter_hz": 400,   # CW filter half-width ±Hz (0 = WIDE); default = classic ±400
         "delivery_pct": 0.0}   # capture-integrity dial: samples/(wall*fs), the no-lie gauge
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
BUILD = "0804-jitterbuf"   # bump on UI changes: an OPEN tab keeps running its
#   old JS across panel restarts, silently - the page compares this via /state
#   and tells the user to refresh (8/04: six deploys, user tested stale code)
AUDIO = deque(maxlen=AUD_FS * 45)      # streaming QUEUE - the wav handler DRAINS it
EARS_RING = deque(maxlen=AUD_FS * 45)  # rolling 45 s HISTORY for the ears lane -
#   separate from AUDIO on purpose: the streamer clears AUDIO every 50 ms, so a
#   listener starves anything else reading it (found 8/04: ears lane read
#   "filling 1058/120960" - and the drain also explains "barely audible":
#   the stream pads silence whenever the queue runs dry)
SIGLIST = {"band": None, "sigs": []}   # classifier's authoritative carrier list: [{khz,snr,cw,wpm}]
SCANNING = False                       # True while an all-bands scan is hopping (prevents overlap)
SCANRES = {"running": False, "done": False, "results": [], "best": None,
           "at": None}                 # async scan progress/results, served in /state
_lock = threading.Lock()
_alock = threading.Lock()
_win = np.hanning(N_FFT).astype(np.float32)
_win_hi = np.hanning(N_HI).astype(np.float32)
_lp = firwin(159, 1500.0 / (FS / 2)).astype(np.float32)   # audio CW filter
_narrow8k = firwin(129, 400.0 / 4000.0).astype(np.float32)  # ±400 Hz single-station filter @8 kHz

# ── user-selectable CW filter (EXP-7: a CENTERED narrow filter = ~6× QRM rejection).
# Half-widths in ±Hz; 0 = WIDE (no narrow stage). Decode + audio both honor it.
CW_FILTER_CHOICES = (0, 400, 250, 150)
_dec_taps = {400: _narrow8k}            # decode-path complex LP taps @8 kHz, per bw
_bp_taps = {}                           # audio bandpass taps @ audio rate, per bw


def _decode_taps(bw):
    t = _dec_taps.get(bw)
    if t is None:
        t = firwin(129, bw / 4000.0).astype(np.float32)
        _dec_taps[bw] = t
    return t


def _audio_bp(bw):
    """Audio-rate bandpass BFO±bw — the 'tune your ear in' filter."""
    t = _bp_taps.get(bw)
    if t is None:
        nyq = AUD_FS / 2.0
        lo = max(60.0, BFO_HZ - bw) / nyq
        hi = min(nyq * 0.9, BFO_HZ + bw) / nyq
        t = firwin(101, [lo, hi], pass_zero=False).astype(np.float32)
        _bp_taps[bw] = t
    return t


def envelope_locked(iq, off_hz, aud=8000, bw=None):
    """Isolate ONE station: shift its carrier to DC, keep only ±bw Hz (rejects
    adjacent CW — EXP-7's ~6× QRM filter), then envelope. bw=None uses the panel's
    selected CW filter (STATE cw_filter_hz); bw=0 = WIDE (no narrow stage).
    NOTE: _decode_taps are designed at 8 kHz — callers must keep aud=8000."""
    from math import gcd
    from scipy.signal import resample_poly
    if bw is None:
        bw = STATE.get("cw_filter_hz", 400)
    n = np.arange(len(iq), dtype=np.float64)
    x = (iq * np.exp(-2j * np.pi * off_hz / FS * n)).astype(np.complex64)
    g = gcd(int(aud), int(FS))
    xr = resample_poly(x, int(aud) // g, int(FS) // g)      # complex -> 8 kHz
    xf = lfilter(_decode_taps(bw), 1.0, xr) if bw else xr   # ±bw one-station filter (WIDE skips)
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


# ── auto-centering carrier lock (the cocktail-party lock, queue item #1) ──
# find_offset picks the strongest POWER near the cursor — in a pileup that locks
# the loudest neighbour, not the copyable CW, and EXP-7 proved a narrow filter
# centered wrong DELETES the wanted signal. Instead: take the power peaks near the
# cursor as candidates, judge each by EYE-OPENING (copyability, like a human ear
# picking the readable fist out of the pile), and HOLD the winner with hysteresis
# so the lock doesn't flap between stations mid-QSO. Re-acquires when the user
# moves the cursor (tune_khz changes).
TRACK = {"off": None, "eye": 0.0, "khz": None}


def _eye_lock(iq, base, search):
    """Pick the carrier offset near `base` with the openest EYE. Returns (off, eye)."""
    s = iq[-int(6 * FS):] if len(iq) > 6 * FS else iq       # recent 6 s: cheap + current
    n = np.arange(len(s), dtype=np.float64)
    x = (s * np.exp(-2j * np.pi * base / FS * n)).astype(np.complex64)
    N = 1 << 14
    m = len(x) // N * N
    if m < N:
        return base + cw.find_offset(x, FS, search), 0.0
    seg = x[:m].reshape(-1, N) * np.hanning(N).astype(np.float32)
    P = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2).mean(0)
    c = N // 2
    k = max(2, int(search / (FS / N)))
    band = P[c - k:c + k]
    noise = np.median(band)
    idx = [i for i in range(1, len(band) - 1)
           if band[i] >= band[i - 1] and band[i] > band[i + 1] and band[i] > 4 * noise]
    idx.sort(key=lambda i: -band[i])
    cands = [0.0] + [(i - k) * FS / N for i in idx[:4]]   # 0.0 = exactly where the user pointed
    if len(cands) == 1:
        cands.append(cw.find_offset(x, FS, search))
    best_off, best_eye = cands[0], -1.0
    for off in cands:                                       # judge by EYE, not power
        env, _ = envelope_locked(s, base + off, bw=400)
        eye = cw_quality.eye_opening(env)[0]
        if eye > best_eye:
            best_off, best_eye = off, eye
    # hysteresis: keep the currently-tracked carrier unless the new winner is
    # CLEARLY better — a mid-QSO key-up must not hand the lock to a neighbour.
    to = TRACK["off"]
    if (TRACK["khz"] == STATE["tune_khz"] and to is not None
            and abs(to - (base + best_off)) > 25):
        env, _ = envelope_locked(s, to, bw=400)
        teye = cw_quality.eye_opening(env)[0]
        if teye >= max(2.0, 0.75 * best_eye):
            TRACK["eye"] = teye
            return to, teye
    TRACK.update(off=base + best_off, eye=best_eye, khz=STATE["tune_khz"])
    return base + best_off, best_eye


def decode_cw(iq):
    # base = where to look: the locked carrier, or the live cursor. The eye-based
    # lock then picks the COPYABLE carrier there (not the loudest) and holds it;
    # the CW filter narrows around that carrier, so the filter is always aimed.
    base = STATE["lock_off"] if STATE["chlock"] else cur_off_hz()
    bw = STATE.get("cw_filter_hz", 400)
    search = min(700.0, float(bw)) if bw else 700.0   # never hunt outside the filter
    off, lock_eye = _eye_lock(iq, base, search)
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
            # the dits-and-dashes the decoder actually heard (educational + shows
            # WHERE copy got shaky). Raw-decode Morse, so it aligns with the raw
            # letters even where the LM later re-segments the English above.
            "morse": info.get("morse", "") if ok else "",
            "tokens": info.get("tokens", [])[:64] if ok else [],
            "lock_eye": round(float(lock_eye), 2),
            "elements": info.get("elements", 0), "offset_hz": round(off, 1), "hint": hint}


# FT8 dial frequencies (kHz) per band — the standard sub-band the whole world
# transmits on. The panel mixes THIS onto baseband, unlike the classic CW cursor.
FT8_DIAL_KHZ = {"160m": 1840.0, "80m": 3573.0, "60m": 5357.0, "40m": 7074.0,
                "30m": 10136.0, "20m": 14074.0, "17m": 18100.0, "15m": 21074.0,
                "12m": 24915.0, "10m": 28074.0, "6m": 50313.0}
_FT8_LAST = {"slot": None}      # slot-dedupe: append a transcript line once per slot


def decode_ft8(iq):
    """FT8 via the jt9 engine-adapter (ft8_live.py). FT8 transmissions start on
    :00/:15/:30/:45 UTC, so we cut the 15 s window aligned to the last completed
    slot that fits in the ring (jt9's own +/-2.4 s dt search covers the residual);
    mix the band's FT8 dial to baseband; run jt9; return the decodes. Only emits a
    transcript line ONCE per slot (the decoder thread re-runs faster than 15 s)."""
    import datetime
    import ft8_live
    band = STATE.get("band", "20m")
    dial = FT8_DIAL_KHZ.get(band)
    empty = {"text": "", "q": 0.0, "conf": 0.0, "wpm": 0, "n": 0,
             "decodes": [], "lines": [], "calls": [], "grids": []}
    if dial is None:
        return {**empty, "offset_hz": 0,
                "hint": f"no FT8 dial for {band} — try 20m/40m/30m/…"}
    off = (dial - STATE["center_khz"]) * 1000.0          # Hz: capture center -> FT8 dial
    fs = FS
    total_s = len(iq) / fs
    now = datetime.datetime.now(datetime.timezone.utc)
    into = (now.second % 15) + now.microsecond / 1e6     # seconds into the current slot
    start_s = total_s - into - 15.0                      # window start, seconds from buffer head
    if start_s >= 0:
        a = int(start_s * fs)
        seg, aligned = iq[a:a + int(15 * fs)], True
    else:
        seg, aligned = iq[-int(15 * fs):], False         # fallback: last 15 s
    if len(seg) < int(14 * fs):
        return {**empty, "offset_hz": round(off),
                "hint": "buffer too short for a 15 s FT8 slot"}
    slot_key = int((now.timestamp() - into) // 15)
    new_slot = slot_key != _FT8_LAST["slot"]
    _FT8_LAST["slot"] = slot_key
    wav = HERE.parent / "lab" / "_ft8_panel.wav"
    wav.parent.mkdir(exist_ok=True)
    ft8_live.iq_to_wav(seg.astype(np.complex64), None, wav, off_hz=off, fs=fs)
    recs, err = ft8_live.decode_wav(wav)
    if err:
        return {**empty, "offset_hz": round(off), "hint": err[:80]}
    calls, grids = [], []
    for r in recs:
        for c in r.get("calls", []):
            if c not in calls:
                calls.append(c)
        if r.get("grid") and r["grid"] not in grids:
            grids.append(r["grid"])
    lines = [f"{r['snr']:+3d} {r['msg']}" for r in recs[:12]]
    text = " / ".join(r["msg"] for r in recs[:8]) if (recs and new_slot) else ""
    hint = "" if recs else "no FT8 decodes this slot (need a live signal on the dial)"
    return {"text": text, "q": 1.0 if recs else 0.0,
            "conf": round(min(1.0, len(recs) / 5.0), 2), "wpm": 0,
            "offset_hz": round(off), "n": len(recs), "decodes": recs[:12],
            "lines": lines, "calls": calls, "grids": grids,
            "aligned": aligned, "hint": hint}


DECODERS = {"CW": decode_cw, "FT8": decode_ft8}

# ── logbook: harvest callsigns like a real ham, and score them ──
LOGFILE = HERE.parent / "lab" / "cw_log.jsonl"
HARVEST_DIR = HERE.parent / "lab" / "cw_harvest"
LOGBOOK = {}                       # call -> record
CALL_RE = re.compile(r"^[A-Z0-9]{1,2}[0-9][A-Z]{1,4}$")
PROSIGN = {"CQ", "DE", "QRL", "QSL", "QSO", "QTH", "QRZ", "QRM", "QRN", "QSB",
           "QRP", "TU", "GM", "GA", "GE", "RST", "AGN", "BK", "AR", "SK", "KN",
           "73", "88", "FB", "OM", "UR", "PSE", "POTA", "SOTA", "WX", "TNX"}


def extract_calls(text, eye=99.0):
    """Callsign-pattern tokens that are confident: repeated (hams send calls
    2-3x) or right after DE/CQ. CONSENSUS-GATED (EXP-9): agreeing repeats are
    real signal-redundancy evidence at any eye; the single-shot after-DE/CQ path
    is only trusted when the eye is OPEN (>=3) — a lone garble following a lucky
    'DE' in noise was the false-callsign source (the old N2TE/W4TT-from-noise)."""
    toks = text.upper().split()
    out = {}
    for i, t in enumerate(toks):
        if t in PROSIGN or not CALL_RE.match(t):
            continue
        conf = 0
        if toks.count(t) >= 2:
            conf += 2                          # >=2 agreeing repeats = consensus
        if i > 0 and toks[i - 1] in ("DE", "CQ"):
            conf += 2 if eye >= 3.0 else 1     # context alone needs an open eye
        if 3 <= len(t) <= 6:
            conf += 1
        if conf >= 2 and (toks.count(t) >= 2 or eye >= 3.0):
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
        # CW filter on the EAR too: audio-rate bandpass BFO±bw (tune your ear in)
        bw = STATE.get("cw_filter_hz", 400)
        if bw:
            if getattr(self, "aud_bw", None) != bw:
                self.aud_bw = bw; self.zi2 = None           # re-init state on width change
            t = _audio_bp(bw)
            if getattr(self, "zi2", None) is None:
                self.zi2 = lfilter_zi(t, 1.0).astype(np.float32) * (a[0] if len(a) else 0.0)
            a, self.zi2 = lfilter(t, 1.0, a, zi=self.zi2)
        pk = float(np.abs(a).max())
        self.agc = max(self.agc * 0.995, pk, 1e-4)
        a16 = np.clip(a / self.agc * 7000.0, -32767, 32767).astype(np.int16)
        with _alock:
            AUDIO.extend(a16)
            EARS_RING.extend(a16)

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
        fails = 0
        # capture-integrity dial (8/03): samples==wall*fs or the decoder is
        # reading OUR seams as keying (W1AW 55 dB decoded to salad; every
        # absurd live wpm was ~96-99 = the chop rate, not an operator).
        dl_t0, dl_got, hb_t = time.time(), 0, 0.0
        while STATE["running"]:
            if radio_lock and radio_lock.should_yield():
                break
            if STATE["center_khz"] != cur:
                cur = STATE["center_khz"]
                sdr.setFrequency(SOAPY_SDR_RX, 0, cur * 1e3)
                ring_clear(); self.zi = None; time.sleep(0.15)
            r = sdr.readStream(st, [buf], 16384, timeoutUs=500000)  # small reads = fast waterfall (~15 rows/s)
            if r.ret <= 0:
                fails += 1
                if fails > 16:            # ~8s of no data = SDR stalled (e.g. band-hop wedge)
                    STATE["err"] = "SDR stalled — reopening"      # break -> run() retries _session (reopen stream)
                    break
                continue
            fails = 0
            dl_got += r.ret
            now = time.time()
            if now - dl_t0 >= 2.0:
                STATE["delivery_pct"] = round(100.0 * dl_got / ((now - dl_t0) * FS), 1)
                dl_t0, dl_got = now, 0
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
            if radio_lock and now - hb_t >= 5.0:
                hb_t = now                 # file I/O every 65 ms stalls the read
                radio_lock.heartbeat()     # loop; TTL is 90 s - 5 s is plenty
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
                                           "text": res["text"], "q": res["q"],
                                           "morse": res.get("morse", "")})
                        snr = round(max(0, SPEC["peak_db"] - SPEC["noise_db"]), 1)
                if res.get("text") and res["mode"] == "CW":
                    got = log_calls(extract_calls(res["text"], eye=res.get("eye_q", 0.0)), STATE["band"],
                                    STATE["center_khz"], snr)
                    if got:
                        res["new_calls"] = got         # log_calls returns a list of call strings
                        with _lock:
                            DECODE["new_calls"] = got
                elif res.get("text") and res["mode"] == "FT8" and res.get("calls"):
                    # FT8 carries verified callsigns directly (structured) - log them
                    # once per slot (text is only set on a new slot, so this dedupes).
                    got = log_calls(res["calls"], STATE["band"], STATE["center_khz"], snr)
                    if got:
                        res["new_calls"] = got
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
            # 20 s of air, not 8 (8/04 live: the user's eyes kept finding
            # fists the 8 s window missed - an op pausing between words
            # doesn't blink enough in 8 s to pass the rhythm gates; eyes
            # integrate the whole waterfall history. 20 s is the window the
            # corpus validation actually proved.)
            iq = ring_snapshot(20)
            if iq is None:
                continue
            band = STATE["band"]
            # candidates = union(whole-band keying map, loud carriers). cw_map
            # fingerprints EVERY bin for Morse rhythm in one pass - it finds the
            # keyed fists that SNR-argmax walks straight past (8/03: the user
            # found them by eye while detect_signals said the band was empty).
            cands = list(detect_signals())
            try:
                for h in cw_map.cw_map(iq, FS, top=12):
                    khz = round(STATE["center_khz"] + h["khz_off"], 3)
                    hit = next((c for c in cands
                                if abs(khz - c["khz"]) <= 0.3), None)
                    if hit is None:
                        cands.append({"khz": khz, "snr": h["snr_db"],
                                      "map": True})
                    else:
                        hit["map"] = True   # loud carrier WITH keying rhythm
            except Exception:
                pass
            # judge each candidate on the FRESHEST 8 s: the 20 s window is for
            # cw_map's rhythm search only - a 20 s eye smears shut under QSB
            # (8/04: 26 carriers, all tagged data/busy, zero badges)
            iq8 = iq[-int(8 * FS):]
            out = []
            for s in cands:
                co = (s["khz"] - STATE["center_khz"]) * 1000.0
                cw_ok, wpm, eye = False, 0, 0.0
                try:
                    ss = iq8[:int(FS)]
                    n = np.arange(len(ss))
                    x = (ss * np.exp(-2j * np.pi * co / FS * n)).astype(np.complex64)
                    off = co + cw.find_offset(x, FS, 400)
                    env, aud = envelope_locked(iq8, off)
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
                # three honest tiers, not two (8/04: the user was LISTENING to
                # partial Morse while zero badges showed): True = the decoder
                # copies it; None = keying RHYTHM seen (cw_map) but not clean
                # copy - a human may still enjoy it, badge it '?'; False = no
                # evidence of Morse at all.
                tier = True if cw_ok else (None if s.get("map") else False)
                out.append({"khz": s["khz"], "snr": s["snr"], "cw": tier,
                            "wpm": wpm, "eye": round(float(eye), 1)})
            # the classifier is the signal-list authority: detect + decode in ONE
            # pass, so tags always match their carrier (no cross-snapshot mismatch)
            with _lock:
                SIGLIST["band"] = band; SIGLIST["sigs"] = out


def _scan_worker():
    """Async all-bands scan body (started by /scanbands, guarded by SCANNING).
    One cw_map whole-band look per band - no per-carrier probing, no server
    lock held, page follows progress via /state's SCANRES."""
    global SCANNING
    orig = (STATE["band"], STATE["center_khz"])
    results = []
    try:
        for band, ctr in BANDS.items():
            SCANRES["at"] = band
            STATE["band"] = band; STATE["center_khz"] = float(ctr)  # reader retunes
            time.sleep(2.2)      # settle (gentle hop rate; fast hops wedge the RSPdx)
            time.sleep(8.0)      # let the ring refill with THIS band's air
            iq = ring_snapshot(8)
            hits = []
            if iq is not None:
                try:
                    hits = cw_map.cw_map(iq, FS, top=8)
                except Exception:
                    pass
            results.append({"band": band, "cw": len(hits),
                            "eye": round(max([h["score"] for h in hits], default=0.0) * 5, 1),
                            "hits": [{"khz": round(ctr + h["khz_off"], 3),
                                      "score": h["score"], "wpm": h["wpm_est"]}
                                     for h in hits[:5]]})
            SCANRES["results"] = list(results)
    except Exception as e:
        STATE["err"] = f"scan: {e}"[:80]
    finally:
        SCANNING = False
    best = max(results, key=lambda r: (r["cw"],
               max([h["score"] for h in r["hits"]], default=0.0))) if results else None
    if best and best["cw"] > 0:
        STATE["band"] = best["band"]; STATE["center_khz"] = float(BANDS[best["band"]])
        STATE["tune_khz"] = best["hits"][0]["khz"]   # straight onto the best fist
        STATE["chlock"] = False
    else:                                     # nothing keyed anywhere -> restore
        STATE["band"], STATE["center_khz"] = orig
    SCANRES.update(running=False, done=True, best=best, at=None)


class EarsDecoder(threading.Thread):
    """The DECODE-WHAT-I-HEAR lane (8/03, user ask): decode the SAME audio
    stream the user's ears get - tone-find, envelope, classic decoder. When
    the IQ lane goes quiet but a human hears Morse, this lane is the referee
    (it read the user's 34 wpm find and W1AW when the live lane showed '')."""
    daemon = True

    def run(self):
        while True:
            time.sleep(25)
            if not STATE["running"] or STATE["mode"] != "CW":
                continue
            try:
                with _alock:
                    pcm = np.array(EARS_RING, np.float64) / 32768.0
                if len(pcm) < AUD_FS * 15:
                    with _lock:                       # breadcrumb, not silence
                        DECODE["ears"] = {"text": "", "q": 0.0, "wpm": 0,
                                          "tone_hz": 0, "elements": 0,
                                          "note": f"filling {len(pcm)}/{AUD_FS*15}"}
                    continue
                spec = np.abs(np.fft.rfft(pcm))
                freqs = np.fft.rfftfreq(len(pcm), 1.0 / AUD_FS)
                m = (freqs > 200) & (freqs < 3000)
                tone = float(freqs[m][np.argmax(spec[m])])
                t = np.arange(len(pcm)) / AUD_FS
                bb = pcm * np.exp(-2j * np.pi * tone * t)
                k = max(1, int(AUD_FS // 200))
                env = np.abs(np.convolve(bb, np.ones(k) / k, "same"))[::max(1, int(AUD_FS // 1000))]
                txt, info = cw.decode_env_auto2(env.astype(np.float32), 1000.0)
                w = float(info.get("wpm", 0))
                # salad gate (8/04 gauntlet finding): a strong STEADY carrier
                # decodes to ?/M/T/E-heavy salad with big element counts - 74
                # of 81 overnight "ears copies" were this. Only surface text
                # whose letter mix looks like language, and always report q.
                chars = [c for c in txt if c != " "]
                q = (1.0 - sum(1 for c in chars if c in "?MTE") / len(chars)) \
                    if chars else 0.0
                ears = {"text": txt[-160:] if (3 <= w <= 45 and q > 0.45) else "",
                        "q": round(q, 2), "wpm": round(w, 1),
                        "tone_hz": round(tone),
                        "elements": int(info.get("elements", 0))}
                with _lock:
                    DECODE["ears"] = ears
            except Exception as e:
                # a silent forever-failure hid this lane for a whole session
                # (8/04) - one honest breadcrumb per fault, never a pass
                import traceback
                with open(HERE.parent / "lab" / "ears_fault.log", "a",
                          encoding="utf-8") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} {e!r}\n"
                            f"{traceback.format_exc()}\n")


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
                     "lock", "err", "chlock", "cw_filter_hz",
                     "delivery_pct")}, "span": SPAN_KHZ,
                    "decode": dict(DECODE), "bands": BANDS, "modes": MODES,
                    "transcript": list(TRANSCRIPT)[-14:], "scan": dict(SCANRES),
                    "build": BUILD,
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
        elif u.path == "/cwfilter":            # CW filter half-width: wide/400/250/150 (±Hz)
            try:
                hz = q.get("hz", ["400"])[0]
                v = 0 if hz in ("wide", "0") else int(hz)
                if v in CW_FILTER_CHOICES:
                    STATE["cw_filter_hz"] = v
            except ValueError:
                pass
            self._send(json.dumps({"ok": True, "cw_filter_hz": STATE["cw_filter_hz"]}))
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
        elif u.path == "/scanbands":
            # ASYNC all-bands scan (8/03): the synchronous version held every
            # request behind a ~50 s hop sweep; a second press froze the UI for
            # good. Now: start the guarded worker, return at once, the page
            # polls /state for SCANRES. Per band: one settle + ring fill, then
            # ONE cw_map whole-band look (every bin fingerprinted for keying
            # rhythm) instead of eye-probing the 6 loudest carriers.
            global SCANNING
            if SCANNING:
                self._send(json.dumps({"started": False, "busy": True})); return
            SCANNING = True
            SCANRES.update(running=True, done=False, results=[], best=None,
                           at=STATE["band"])
            threading.Thread(target=_scan_worker, daemon=True).start()
            self._send(json.dumps({"started": True}))
        elif u.path == "/tune":                # move the CURSOR (hops window if needed)
            try:
                khz = float(q["khz"][0])
                # cross-band click (8/04, user: "the scan found Morse but I
                # can't tune into any of it"): a scan hit or stale signal-list
                # row can be OUTSIDE the current window - clamping silently
                # pinned the cursor at the edge. Now the window HOPS to the
                # band that contains the target, so a click always lands.
                out = abs(khz - STATE["center_khz"]) > SPAN_KHZ / 2 - 0.5
                if out:
                    for _b, _ctr in BANDS.items():
                        if abs(khz - float(_ctr)) <= SPAN_KHZ / 2 - 0.5:
                            STATE["band"] = _b
                            STATE["center_khz"] = float(_ctr)  # reader retunes
                            break
                # snap=1: refine to the TRUE peak in the high-res (~30 Hz) spectrum
                # near the click, so the cursor lands exactly on the signal and stays
                # locked to it at any zoom (coarse display bins otherwise drift on zoom)
                # (skipped on a cross-band hop: the hi-res slice is the OLD band)
                if not out and q.get("snap", ["0"])[0] == "1":
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
            # jitter buffer (8/04, user: "a constant skip" during static): the
            # old loop drained EVERYTHING every 50 ms and padded silence the
            # moment production jitter left the queue empty - a ~15 Hz comb of
            # gaps, most audible against noise. Now: build a 300 ms prebuffer,
            # send fixed 100 ms chunks only when real audio is available, and
            # pad only after a genuine gap (>=400 ms) to keep the player alive.
            ch = AUD_FS // 10
            with _alock:
                AUDIO.clear()                      # fresh start, no stale backlog
            t0 = time.time()
            while STATE["running"] and time.time() - t0 < 3.0:
                with _alock:
                    if len(AUDIO) >= 3 * ch:
                        break
                time.sleep(0.05)
            starve = 0
            while STATE["running"]:
                with _alock:
                    n = len(AUDIO)
                    take = min(n, 3 * ch) if n >= ch else 0
                    chunk = (np.array([AUDIO.popleft() for _ in range(take)],
                                      np.int16).tobytes() if take else b"")
                if chunk:
                    starve = 0
                    self.wfile.write(chunk)
                else:
                    starve += 1
                    if starve >= 8:
                        self.wfile.write(b"\x00\x00" * ch)
                time.sleep(0.05)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8647)
    ap.add_argument("--antenna", default=STATE["antenna"],
                    help='RSPdx port, e.g. "Antenna A" (loop, HF) / "Antenna C" (discone)')
    args = ap.parse_args()
    STATE["antenna"] = args.antenna
    os.environ["PATH"] = r"C:\Program Files\SDRplay\API\x64" + os.pathsep + os.environ.get("PATH", "")
    _load_log()
    SDRWorker().start()
    Decoder().start()
    Verifier().start()
    Classifier().start()
    EarsDecoder().start()
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
.sigbadge{position:absolute;top:2px;transform:translateX(-50%);z-index:6;background:#0e2f16;color:#7dff9a;border:1px solid #2fa15a;border-radius:6px;padding:1px 7px;font-size:11px;line-height:1.5;cursor:pointer;white-space:nowrap;box-shadow:0 0 5px #0008}
.sigbadge:hover{background:#17512a}
.sigbadge.cand{color:#9aa;border-color:#456;background:#101820}
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
.advband.good{border-color:var(--good);color:var(--good)}
.scanb{background:#1a2740;color:#9db8ff;border:1px solid #2a3f6a;width:100%;padding:9px;font-size:12px;font-weight:600;margin-top:4px}
.scanb:hover{background:#22335a}.scanb:disabled{opacity:.6;cursor:wait}
.listen{width:100%;padding:9px;font-size:13px;font-weight:700}.listen.on{background:var(--acc2);color:#1a0409;border-color:var(--acc2)}
.lockb{width:100%;padding:9px;font-size:13px;font-weight:700}.lockb.on{background:var(--warn);color:#1a1204;border-color:var(--warn)}
.fbtn{flex:1;font-size:12px;padding:7px 4px}.fbtn.on{background:#123c49;color:#8fe0f0;border-color:#2e7a8f;font-weight:700}
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
  <div class=left><canvas id=spec></canvas><canvas id=wf></canvas><div class=curline id=curline></div><div id=badges></div>
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
    <button class=autob id=autob onclick=autotune()>&#9673; AUTO-TUNE (this band)</button>
    <button class=scanb id=scanb onclick=scanBands()>&#128270; SCAN ALL BANDS for CW</button>
    <div class=advout id=scanout></div>
    <button class=lockb id=lockb onclick=togLock()>&#128275; LOCK channel</button>
    <div>
      <div class=lbl style="display:flex;justify-content:space-between;align-items:center">
        <span>CW signals on air</span>
        <span><button class=step onclick="step(-1)">&#9664;</button> <button class=step onclick="step(1)">&#9654;</button></span>
      </div>
      <div class=siglist id=siglist></div>
    </div>
    <button class=listen id=listenb onclick=togListen()>&#9654; LISTEN (live audio)</button>
    <div class=lbl style="margin-top:6px" title="Narrow the receiver around the tuned signal — rejects neighbouring CW (decode + audio)">CW FILTER &plusmn;Hz</div>
    <div style="display:flex;gap:5px">
      <button class=fbtn id=fb0 onclick=setFilt(0)>WIDE</button>
      <button class=fbtn id=fb400 onclick=setFilt(400)>400</button>
      <button class=fbtn id=fb250 onclick=setFilt(250)>250</button>
      <button class=fbtn id=fb150 onclick=setFilt(150)>150</button>
    </div>
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
    <div><div class=lbl id=declbl>Live Morse transcript</div><div class=xscript id=xscript></div>
    <div id=earsline style="margin-top:6px;padding:8px;border:1px dashed var(--hair);border-radius:8px;font-size:14px"></div></div>
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
  $('route').textContent=(d.route?({neural:'🧠 neural AI',mf:'matched-filter (fading)',mf2:'matched-filter',classic:'classic'}[d.route]||d.route):'—')+(d.lock_eye>0?(' · lock eye '+d.lock_eye.toFixed(1)):'');
  const fhz=(ST.cw_filter_hz===undefined)?400:ST.cw_filter_hz;
  for(const v of [0,400,250,150]){const b=$('fb'+v);if(b)b.classList.toggle('on',fhz===v);}
  $('wpm').textContent=d.wpm?d.wpm.toFixed(1):'—';
  $('sm').textContent=(ST.smeter||0).toFixed(0)+' dB';$('smbar').style.width=Math.min(100,(ST.smeter||0)*2.2)+'%';
  $('declbl').textContent=ST.mode==='CW'?'Live Morse transcript':ST.mode+' decode';
  const xs=$('xscript');
  if(ST.mode==='CW'){
    const tr=ST.transcript||[];    // rolling history so intermittent copy accumulates & stays visible
    const mstyle='display:block;font-family:monospace;opacity:.5;font-size:.8em;letter-spacing:2px;margin-bottom:4px';
    xs.innerHTML = tr.length ? tr.map(x=>`<span class=xline>${x.text}${(ST.showmorse!==false&&x.morse)?`<span style="${mstyle}">${x.morse}</span>`:''} </span>`).join('')
                  : (d.text?`<span class=xline>${d.text}</span>`:'<div class=sub>…listening for CW…</div>');
    xs.scrollTop=xs.scrollHeight;
  } else xs.innerHTML='<div class=sub>'+ST.mode+' decode coming soon — spectrum + audio live</div>';
  // EARS lane: decodes exactly the audio you are hearing (the referee when
  // the live lane shows nothing but your ears clearly copy Morse)
  const e=d.ears||{};
  $('earsline').innerHTML=(e.text&&e.elements>=20)
    ?`<b>👂 what you're hearing:</b> ${e.text} <span class=sub>(~${e.wpm} wpm, tone ${e.tone_hz} Hz)</span>`
    :'<span class=sub>👂 ears lane: no readable Morse in the audio right now</span>';
  $('hint').textContent=d.hint||'';
  // stale-tab guard: an open tab keeps its old JS across panel restarts
  if(!window._b1)window._b1=ST.build;
  else if(ST.build&&ST.build!==window._b1)
    $('hint').textContent='⟳ the panel was updated behind this tab — press Ctrl+Shift+R to load the new version';
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
async function scanBands(){
  const b=$('scanb'),lbl=b.textContent,o=$('scanout');
  b.disabled=true;o.innerHTML='<span class=sub>looking at every band…</span>';
  try{const st=await api('/scanbands');if(!st.started&&!st.busy)throw 0;}
  catch(e){b.disabled=false;b.textContent=lbl;return;}
  // async scan: the worker hops bands; we follow its progress via /state.scan
  let r={};for(let i=0;i<60;i++){await new Promise(z=>setTimeout(z,2000));
    try{r=(await api('/state')).scan||{};}catch(e){continue;}
    b.textContent=r.at?('… looking at '+r.at+' …'):lbl;
    const rows=(r.results||[]).map(x=>`<span class="advband${x.cw>0?' good':''}" onclick="set('band='+'${x.band}')">${x.band} ${x.cw>0?('✓'+x.cw):'—'}</span>`).join('');
    if(rows)o.innerHTML=`<div class=advrow>${rows}</div>`;
    if(r.done)break;}
  await refresh();
  // every found fist is a BUTTON: click = jump straight to it, even across
  // bands (the /tune window-hop makes cross-band clicks just work)
  const fists=(r.results||[]).flatMap(x=>(x.hits||[]).map(h=>
    `<span class="advband good" onclick="tune(${h.khz})" title="score ${h.score}">${x.band} ${h.khz.toFixed(1)} ~${h.wpm}wpm</span>`)).join('');
  o.innerHTML=(o.innerHTML||'')+
    (fists?`<div class=sub>found Morse — click to listen:</div><div class=advrow>${fists}</div>`
      :'<div class=sub>no keyed Morse on any band right now — try again this evening</div>');
  b.disabled=false;b.textContent=lbl;}
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
async function setFilt(hz){await api('/cwfilter?hz='+hz);refresh();}
let SIGS=[];   // latest classified signals - feeds the waterfall badges
// Waterfall badges (8/04, user ask): a clickable tab floats right above each
// detected Morse signal ON the waterfall - see code, click code, hear code.
function renderBadges(){const bd=$('badges');if(!bd)return;
  if(VC===null||VS===null){bd.innerHTML='';bd._key='';return;}
  const w=spec.clientWidth;
  const vis=SIGS.filter(x=>x.cw!==false);
  // STABLE DOM (8/04, the real click bug): this runs from the draw loop
  // several times a second - rebuilding innerHTML each pass destroys the
  // badge mid-press, so human clicks mostly land on a corpse. Rebuild ONLY
  // when the signal set changes; otherwise just slide the existing badges.
  const key=vis.map(x=>x.khz.toFixed(2)+':'+x.cw).join('|');
  if(key===bd._key){
    for(const el of bd.children){
      const px=(parseFloat(el.dataset.khz)-(VC-VS/2))/VS*w;
      el.style.left=px.toFixed(0)+'px';
      el.style.display=(px<14||px>w-14)?'none':'';}
    return;}
  bd._key=key;
  bd.innerHTML=vis.map(x=>{
    const px=(x.khz-(VC-VS/2))/VS*w;
    const cls=x.cw===true?'sigbadge':'sigbadge cand';
    const txt=x.cw===true?('&#9679; CW '+(x.wpm||'')):'?';
    return `<span class="${cls}" data-khz="${x.khz}" style="left:${px.toFixed(0)}px;${(px<14||px>w-14)?'display:none;':''}" `+
      `title="${x.khz.toFixed(2)} kHz &middot; ${x.snr||'?'} dB - click to listen" `+
      `onmousedown="event.stopPropagation();event.preventDefault();tune(${x.khz});">${txt}</span>`;}).join('');}
async function pollSignals(){let s;try{s=await api('/signals');}catch(e){return;}
  const list=$('siglist'),sigs=s.signals||[],c=s.center;
  SIGS=sigs;renderBadges();
  // copyable CW floats to the TOP of the list (8/04 user: had to scroll to
  // the bottom to find the one illuminated row); then candidates, data last
  sigs.sort((a,b)=>((b.cw===true)-(a.cw===true))||((a.cw===false)-(b.cw===false))||(b.eye||0)-(a.eye||0));
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
  renderBadges();   // badges track every zoom/pan so they stay glued to their signals
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
  // CW-filter passband shading ±bw around the cursor — SEE what the radio hears
  const bwhz=ST.cw_filter_hz;
  if(bwhz&&tk&&VS){const bk=bwhz/1000;
    const x1=(tk-bk-(VC-VS/2))/VS*w, x2=(tk+bk-(VC-VS/2))/VS*w;
    if(x2>0&&x1<w){const a1=Math.max(0,x1),a2=Math.min(w,x2);
      sx.fillStyle='rgba(255,216,74,.09)';sx.fillRect(a1,0,a2-a1,h);
      sx.strokeStyle='rgba(255,216,74,.35)';sx.strokeRect(a1+.5,.5,a2-a1-1,h-1);}}
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
