#!/usr/bin/env python3
"""panel.py - hamTuna adaptive SDR panel (SDRuno-style, in the browser).

The Tuna thesis, made visible: a real spectrum + waterfall receiver like
SDRuno/SDRplay, but with a TRUTH DIAL — the software surfaces how well the
active mode is decoding and closes the loop (auto-find the signal, self-
calibrate, show confidence). Every mode plugs into one registry so "add a
ham mode" == "add a decoder function".

  python tools/panel.py            # http://localhost:8647
  python tools/panel.py --port N

v1 modes: CW live-decode + truth dial (SSB/AM/APRS/FT8 = spectrum-only stubs,
wired for the next decoders). Single SDR via radio_lock@80. Antenna C (HF).
"""
import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                      # cw.py
sys.path.insert(0, r"Z:\src\gr-radiotuna\tools")   # radio_lock
import cw

try:
    import radio_lock
except Exception:
    radio_lock = None

FS = 250_000.0
N_FFT = 2048
DISP_BINS = 500
DECODE_SECS = 12          # IQ accumulated per decode pass
SPAN_KHZ = FS / 1e3       # 250 kHz visible span

# band -> default CW-segment center (kHz)
BANDS = {"160m": 1830, "80m": 3560, "40m": 7030, "30m": 10120, "20m": 14030,
         "17m": 18080, "15m": 21030, "12m": 24906, "10m": 28030}
MODES = ["CW", "SSB", "AM", "FM", "APRS", "FT8"]

STATE = {
    "center_khz": 14030.0, "band": "20m", "mode": "CW",
    "ifgr": 30, "rfsel": 0, "running": True, "antenna": "Antenna C",
    "lock": "none", "err": "",
}
SPEC = {"db": [0.0] * DISP_BINS, "peak_db": -120.0, "noise_db": -120.0, "ts": 0.0}
DECODE = {"text": "", "wpm": 0.0, "q": 0.0, "conf": 0.0, "elements": 0,
          "mode": "CW", "ts": 0.0, "hint": ""}
_lock = threading.Lock()
_win = np.hanning(N_FFT).astype(np.float32)


def _spectrum(iq):
    """Welch-ish dB spectrum over the 250 kHz window, decimated to DISP_BINS."""
    n = len(iq) // N_FFT * N_FFT
    if n < N_FFT:
        return None
    seg = iq[:n].reshape(-1, N_FFT) * _win
    p = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2).mean(0)
    db = 10 * np.log10(p + 1e-9)
    # decimate to DISP_BINS by max-pooling (keep carriers visible)
    step = len(db) // DISP_BINS
    db = db[:step * DISP_BINS].reshape(DISP_BINS, step).max(1)
    return db.astype(np.float32)


def decode_cw(iq):
    off = cw.find_offset(iq, FS, search=50000)
    env, aud = cw.envelope(iq, FS, off)
    txt, info = cw.decode_env(env, aud)
    chars = [c for c in txt if c != " "]
    q = round(sum(1 for c in chars if c != "?") / len(chars), 3) if chars else 0.0
    wpm = info.get("wpm", 0.0)
    ok = 3 <= wpm <= 45 and txt.strip()
    conf = round(q * min(1.0, len(chars) / 12), 3) if ok else 0.0
    hint = "" if ok else "no readable CW here — try Auto-Tune or another band"
    return {"text": txt if ok else "", "wpm": wpm, "q": q, "conf": conf,
            "elements": info.get("elements", 0), "offset_hz": round(off, 1),
            "hint": hint}


DECODERS = {"CW": decode_cw}   # register more modes here


class SDRWorker(threading.Thread):
    daemon = True

    def run(self):
        while True:
            if not STATE["running"]:
                time.sleep(0.4); continue
            try:
                self._session()
            except Exception as e:
                STATE["err"] = str(e)[:120]
                time.sleep(2.0)

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
        acc = deque(maxlen=int(DECODE_SECS * FS))
        buf = np.empty(2 * 65536, np.int16)
        last_decode = 0.0
        while STATE["running"]:
            if radio_lock and radio_lock.should_yield():
                break
            if STATE["center_khz"] != cur:
                cur = STATE["center_khz"]
                sdr.setFrequency(SOAPY_SDR_RX, 0, cur * 1e3)
                acc.clear(); time.sleep(0.15)
            r = sdr.readStream(st, [buf], 65536, timeoutUs=500000)
            if r.ret <= 0:
                continue
            iq = ((buf[0:2 * r.ret:2].astype(np.float32)
                   + 1j * buf[1:2 * r.ret:2].astype(np.float32)) / 32768.0)
            acc.extend(iq)
            db = _spectrum(iq)
            if db is not None:
                with _lock:
                    SPEC["db"] = db.tolist()
                    SPEC["peak_db"] = float(db.max())
                    SPEC["noise_db"] = float(np.percentile(db, 25))
                    SPEC["ts"] = time.time()
            if (STATE["mode"] in DECODERS and len(acc) >= DECODE_SECS * FS * 0.9
                    and time.time() - last_decode > DECODE_SECS):
                last_decode = time.time()
                iqd = np.fromiter(acc, np.complex64, len(acc))
                try:
                    res = DECODERS[STATE["mode"]](iqd)
                except Exception as e:
                    res = {"text": "", "wpm": 0, "q": 0, "conf": 0,
                           "elements": 0, "hint": f"decode err: {e}"[:80]}
                res["mode"] = STATE["mode"]; res["ts"] = time.time()
                with _lock:
                    DECODE.update(res)
                if radio_lock:
                    radio_lock.heartbeat()
        try:
            sdr.deactivateStream(st); sdr.closeStream(st); del sdr
        except Exception:
            pass
        if radio_lock:
            radio_lock.release("hamtuna_panel")
        STATE["lock"] = "released"


def auto_tune():
    """Adaptive: recenter on the strongest carrier in the window (the tuna move)."""
    with _lock:
        db = np.array(SPEC["db"])
    if not len(db):
        return
    binhz = FS / len(db)
    peak = int(np.argmax(db))
    off_hz = (peak - len(db) / 2) * binhz
    STATE["center_khz"] = round(STATE["center_khz"] + off_hz / 1e3, 2)


# ────────────────────────── web ──────────────────────────
def page():
    return PAGE


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/":
            self._send(page(), "text/html; charset=utf-8")
        elif u.path == "/spectrum":
            with _lock:
                self._send(json.dumps({"db": SPEC["db"], "peak": SPEC["peak_db"],
                    "noise": SPEC["noise_db"], "center": STATE["center_khz"],
                    "span": SPAN_KHZ}))
        elif u.path == "/state":
            with _lock:
                self._send(json.dumps({**{k: STATE[k] for k in
                    ("center_khz", "band", "mode", "ifgr", "running", "lock", "err")},
                    "decode": dict(DECODE), "bands": BANDS, "modes": MODES,
                    "smeter": round(max(0, (SPEC["peak_db"] - SPEC["noise_db"])), 1)}))
        elif u.path == "/set":
            if "band" in q:
                b = q["band"][0]
                if b in BANDS:
                    STATE["band"] = b; STATE["center_khz"] = float(BANDS[b])
            if "center" in q:
                try: STATE["center_khz"] = round(float(q["center"][0]), 2)
                except ValueError: pass
            if "mode" in q and q["mode"][0] in MODES:
                STATE["mode"] = q["mode"][0]
            if "ifgr" in q:
                try: STATE["ifgr"] = int(q["ifgr"][0])
                except ValueError: pass
            if "running" in q:
                STATE["running"] = q["running"][0] == "1"
            self._send(json.dumps({"ok": True}))
        elif u.path == "/autotune":
            auto_tune(); self._send(json.dumps({"ok": True, "center": STATE["center_khz"]}))
        else:
            self.send_response(404); self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8647)
    args = ap.parse_args()
    os.environ["PATH"] = r"C:\Program Files\SDRplay\API\x64" + os.pathsep + os.environ.get("PATH", "")
    SDRWorker().start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"hamTuna panel: http://localhost:{args.port}", flush=True)
    srv.serve_forever()


PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<title>hamTuna</title><style>
:root{--bg:#070b10;--panel:#0e1620;--ink:#d6e6f2;--mut:#5f7893;--acc:#2ee6c8;--acc2:#ff5d73;--hair:#182634;--good:#3ad17a;--warn:#f0b23a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:ui-monospace,Consolas,monospace;overflow:hidden}
.top{display:flex;align-items:center;gap:14px;padding:8px 14px;border-bottom:1px solid var(--hair);background:var(--panel)}
.logo{font-weight:700;letter-spacing:.06em;color:var(--acc);font-size:18px}.logo b{color:var(--acc2)}
.freq{font-size:30px;font-weight:700;letter-spacing:.04em;color:#fff;text-shadow:0 0 12px rgba(46,230,200,.35)}
.freq small{font-size:13px;color:var(--mut)}
.sub{color:var(--mut);font-size:12px}
.wrap{display:grid;grid-template-columns:1fr 300px;height:calc(100vh - 52px)}
.left{display:flex;flex-direction:column;min-width:0}
#spec{background:#040709;flex:0 0 200px;width:100%}#wf{background:#000;flex:1;width:100%}
.side{border-left:1px solid var(--hair);background:var(--panel);padding:12px;overflow-y:auto;display:flex;flex-direction:column;gap:14px}
.row{display:flex;flex-wrap:wrap;gap:6px}
button{background:#12202e;color:var(--ink);border:1px solid var(--hair);border-radius:7px;padding:7px 10px;font-family:inherit;font-size:12px;cursor:pointer}
button:hover{border-color:var(--acc)}button.on{background:var(--acc);color:#04110e;border-color:var(--acc);font-weight:700}
button.mode.on{background:var(--acc2);color:#1a0409;border-color:var(--acc2)}
.lbl{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);margin-bottom:5px}
.dial{background:#0a121b;border:1px solid var(--hair);border-radius:10px;padding:12px}
.gauge{height:10px;background:#0a1a16;border-radius:6px;overflow:hidden;margin:6px 0}
.gfill{height:100%;background:linear-gradient(90deg,var(--acc2),var(--warn),var(--good));transition:width .3s}
.big{font-size:22px;font-weight:700}.decode{background:#050b0a;border:1px solid var(--hair);border-radius:8px;padding:10px;min-height:70px;font-size:14px;color:var(--acc);line-height:1.5;word-break:break-word}
.stat{display:flex;justify-content:space-between;font-size:12px;color:var(--mut);padding:2px 0}.stat b{color:var(--ink)}
.autob{background:var(--acc);color:#04110e;font-weight:700;width:100%;padding:10px;font-size:13px}
.smeter{height:8px;background:#0a1a16;border-radius:5px;overflow:hidden}.sfill{height:100%;background:var(--acc);transition:width .2s}
.chip{font-size:10px;padding:2px 7px;border-radius:20px;border:1px solid var(--hair);color:var(--mut)}
.chip.held{color:var(--good);border-color:var(--good)}.chip.busy{color:var(--warn);border-color:var(--warn)}
</style></head><body>
<div class=top>
  <div class=logo>ham<b>Tuna</b></div>
  <div class=freq id=freq>14030.00<small> kHz</small></div>
  <div class=sub id=bandlbl>20m &middot; CW</div>
  <div style=flex:1></div>
  <span class="chip" id=lock>lock</span>
  <div class=sub>TRUTH DIAL &darr; the software knows how well it hears</div>
</div>
<div class=wrap>
  <div class=left><canvas id=spec></canvas><canvas id=wf></canvas></div>
  <div class=side>
    <div><div class=lbl>Band</div><div class=row id=bands></div></div>
    <div><div class=lbl>Mode</div><div class=row id=modes></div></div>
    <button class=autob onclick=autotune()>&#9673; AUTO-TUNE to strongest</button>
    <div class=dial>
      <div class=lbl>Truth Dial &mdash; decode confidence</div>
      <div class=big id=conf>0%</div>
      <div class=gauge><div class=gfill id=confbar style=width:0%></div></div>
      <div class=stat><span>WPM</span><b id=wpm>&mdash;</b></div>
      <div class=stat><span>char quality</span><b id=q>&mdash;</b></div>
      <div class=stat><span>S-meter</span><b id=sm>&mdash;</b></div>
      <div class=smeter><div class=sfill id=smbar style=width:0%></div></div>
    </div>
    <div><div class=lbl id=declbl>CW decode (live)</div><div class=decode id=decode>&hellip;</div></div>
    <div class=sub id=hint></div>
  </div>
</div>
<script>
let ST={};const $=id=>document.getElementById(id);
const spec=$('spec'),sx=spec.getContext('2d'),wf=$('wf'),wx=wf.getContext('2d');
function fit(){spec.width=spec.clientWidth;spec.height=spec.clientHeight;wf.width=wf.clientWidth;wf.height=wf.clientHeight;}
addEventListener('resize',fit);fit();
async function api(p){return (await fetch(p)).json();}
async function refresh(){
  ST=await api('/state');
  $('freq').innerHTML=ST.center_khz.toFixed(2)+'<small> kHz</small>';
  $('bandlbl').textContent=ST.band+' · '+ST.mode;
  const lk=$('lock');lk.textContent=ST.lock;lk.className='chip '+ST.lock;
  // bands
  if(!$('bands').dataset.f){$('bands').dataset.f=1;
    for(const b in ST.bands){const el=document.createElement('button');el.textContent=b;el.onclick=()=>set('band='+b);el.dataset.b=b;$('bands').appendChild(el);}
    ST.modes.forEach(m=>{const el=document.createElement('button');el.className='mode';el.textContent=m;el.onclick=()=>set('mode='+m);el.dataset.m=m;$('modes').appendChild(el);});
  }
  [...$('bands').children].forEach(e=>e.classList.toggle('on',e.dataset.b===ST.band));
  [...$('modes').children].forEach(e=>e.classList.toggle('on',e.dataset.m===ST.mode));
  const d=ST.decode||{};
  $('conf').textContent=Math.round((d.conf||0)*100)+'%';
  $('confbar').style.width=Math.round((d.conf||0)*100)+'%';
  $('wpm').textContent=d.wpm?d.wpm.toFixed(1):'—';
  $('q').textContent=d.q?d.q.toFixed(2):'—';
  $('sm').textContent=(ST.smeter||0).toFixed(0)+' dB';
  $('smbar').style.width=Math.min(100,(ST.smeter||0)*2.2)+'%';
  $('declbl').textContent=ST.mode+' decode (live)';
  $('decode').textContent=d.text?d.text:(ST.mode==='CW'?'…listening…':ST.mode+' decode coming soon — spectrum live');
  $('hint').textContent=d.hint||'';
}
async function set(kv){await api('/set?'+kv);refresh();}
async function autotune(){await api('/autotune');refresh();}
spec.onclick=e=>{const r=spec.getBoundingClientRect();const frac=(e.clientX-r.left)/r.width;
  const c=ST.center_khz,span=ST.mode?250:250;set('center='+(c-span/2+frac*span).toFixed(2));};
// spectrum + waterfall
let WF=[];
async function draw(){
  const s=await api('/spectrum');const db=s.db;if(!db.length){requestAnimationFrame(()=>setTimeout(draw,120));return;}
  const w=spec.width,h=spec.height;sx.clearRect(0,0,w,h);
  // grid
  sx.strokeStyle='#0f2029';sx.lineWidth=1;for(let i=0;i<=4;i++){const y=h*i/4;sx.beginPath();sx.moveTo(0,y);sx.lineTo(w,y);sx.stroke();}
  const lo=s.noise-8,hi=s.peak+6,rng=Math.max(6,hi-lo);
  sx.strokeStyle='#2ee6c8';sx.lineWidth=1.4;sx.beginPath();
  for(let i=0;i<db.length;i++){const x=i/db.length*w,y=h-(db[i]-lo)/rng*h;i?sx.lineTo(x,y):sx.moveTo(x,y);}sx.stroke();
  // center line
  sx.strokeStyle='rgba(255,93,115,.6)';sx.beginPath();sx.moveTo(w/2,0);sx.lineTo(w/2,h);sx.stroke();
  // waterfall row
  const cw_=wf.width,ch=wf.height;const img=wx.getImageData(0,0,cw_,ch);wx.putImageData(img,0,1);
  const row=wx.createImageData(cw_,1);
  for(let x=0;x<cw_;x++){const i=Math.floor(x/cw_*db.length);let v=(db[i]-lo)/rng;v=Math.max(0,Math.min(1,v));
    const c=viridis(v);row.data[x*4]=c[0];row.data[x*4+1]=c[1];row.data[x*4+2]=c[2];row.data[x*4+3]=255;}
  wx.putImageData(row,0,0);
  setTimeout(draw,150);
}
function viridis(t){const r=Math.max(0,Math.min(255,Math.round(255*(0.28+2.2*t-1.6*t*t))));
  const g=Math.round(255*Math.min(1,0.02+1.05*t));const b=Math.round(255*(0.35+0.9*t-1.1*t*t));return[r,g,Math.max(30,b)];}
refresh();setInterval(refresh,1500);draw();
</script></body></html>"""

if __name__ == "__main__":
    main()
