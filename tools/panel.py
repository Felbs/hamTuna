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
try:
    import radio_lock
except Exception:
    radio_lock = None

FS = 250_000.0
N_FFT = 2048
DISP_BINS = 500
DECODE_SECS = 6          # rolling decode window (shorter = more "live")
SPAN_KHZ = FS / 1e3
AUD_DEC = 31             # 250000/31 = 8064.5 Hz audio
AUD_FS = int(FS / AUD_DEC)
BFO_HZ = 600.0          # CW carrier is mixed to this pitch

BANDS = {"160m": 1830, "80m": 3560, "40m": 7030, "30m": 10120, "20m": 14030,
         "17m": 18080, "15m": 21030, "12m": 24906, "10m": 28030}
MODES = ["CW", "SSB", "AM", "FM", "APRS", "FT8"]

STATE = {"center_khz": 14030.0, "band": "20m", "mode": "CW", "ifgr": 30,
         "rfsel": 0, "running": True, "antenna": "Antenna C", "lock": "none", "err": ""}
SPEC = {"db": [0.0] * DISP_BINS, "peak_db": -120.0, "noise_db": -120.0, "ts": 0.0}
DECODE = {"text": "", "wpm": 0.0, "q": 0.0, "conf": 0.0, "elements": 0,
          "mode": "CW", "ts": 0.0, "hint": "", "offset_hz": 0.0}
TRANSCRIPT = deque(maxlen=60)
AUDIO = deque(maxlen=AUD_FS * 4)
_lock = threading.Lock()
_alock = threading.Lock()
_win = np.hanning(N_FFT).astype(np.float32)
_lp = firwin(159, 1500.0 / (FS / 2)).astype(np.float32)   # narrow CW filter


def _spectrum(iq):
    n = len(iq) // N_FFT * N_FFT
    if n < N_FFT:
        return None
    seg = iq[:n].reshape(-1, N_FFT) * _win
    p = (np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2).mean(0)
    db = 10 * np.log10(p + 1e-9)
    step = len(db) // DISP_BINS
    return db[:step * DISP_BINS].reshape(DISP_BINS, step).max(1).astype(np.float32)


def decode_cw(iq):
    off = cw.find_offset(iq, FS, search=50000)
    env, aud = cw.envelope(iq, FS, off)
    txt, info = cw.decode_env(env, aud)
    chars = [c for c in txt if c != " "]
    q = round(sum(1 for c in chars if c != "?") / len(chars), 3) if chars else 0.0
    wpm = info.get("wpm", 0.0)
    ok = 3 <= wpm <= 45 and txt.strip()
    conf = round(q * min(1.0, len(chars) / 10), 3) if ok else 0.0
    return {"text": txt if ok else "", "wpm": wpm, "q": q, "conf": conf,
            "elements": info.get("elements", 0), "offset_hz": round(off, 1),
            "hint": "" if ok else "no readable CW — click a signal on the waterfall or Auto-Tune"}


DECODERS = {"CW": decode_cw}


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
        mixf = BFO_HZ - self.cw_off            # bring carrier to BFO pitch
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
        acc = deque(maxlen=int(DECODE_SECS * FS))
        buf = np.empty(2 * 65536, np.int16)
        last_decode = 0.0
        last_off = 0.0
        while STATE["running"]:
            if radio_lock and radio_lock.should_yield():
                break
            if STATE["center_khz"] != cur:
                cur = STATE["center_khz"]
                sdr.setFrequency(SOAPY_SDR_RX, 0, cur * 1e3)
                acc.clear(); self.zi = None; time.sleep(0.15)
            r = sdr.readStream(st, [buf], 65536, timeoutUs=500000)
            if r.ret <= 0:
                continue
            iq = ((buf[0:2 * r.ret:2].astype(np.float32)
                   + 1j * buf[1:2 * r.ret:2].astype(np.float32)) / 32768.0).astype(np.complex64)
            acc.extend(iq)
            db = _spectrum(iq)
            if db is not None:
                with _lock:
                    SPEC["db"] = db.tolist(); SPEC["peak_db"] = float(db.max())
                    SPEC["noise_db"] = float(np.percentile(db, 25)); SPEC["ts"] = time.time()
            # track CW carrier ~1.5 Hz for the audio BFO
            if STATE["mode"] == "CW" and time.time() - last_off > 1.5 and len(acc) > FS:
                last_off = time.time()
                try:
                    self.cw_off = cw.find_offset(np.fromiter(list(acc)[-int(FS):], np.complex64, int(FS)), FS, 50000)
                except Exception:
                    pass
            if STATE["mode"] == "CW":
                self._audio(iq)
            if (STATE["mode"] in DECODERS and len(acc) >= DECODE_SECS * FS * 0.9
                    and time.time() - last_decode > DECODE_SECS):
                last_decode = time.time()
                iqd = np.fromiter(acc, np.complex64, len(acc))
                try:
                    res = DECODERS[STATE["mode"]](iqd)
                except Exception as e:
                    res = {"text": "", "wpm": 0, "q": 0, "conf": 0, "elements": 0,
                           "hint": f"decode err: {e}"[:80], "offset_hz": 0}
                res["mode"] = STATE["mode"]; res["ts"] = time.time()
                with _lock:
                    DECODE.update(res)
                    if res["text"]:
                        TRANSCRIPT.append({"ts": time.strftime("%H:%M:%S"),
                                           "text": res["text"], "q": res["q"]})
                if radio_lock:
                    radio_lock.heartbeat()
        try:
            sdr.deactivateStream(st); sdr.closeStream(st); del sdr
        except Exception:
            pass
        if radio_lock:
            radio_lock.release("hamtuna_panel")
        STATE["lock"] = "released"


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
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/":
            self._send(PAGE, "text/html; charset=utf-8")
        elif u.path == "/spectrum":
            with _lock:
                self._send(json.dumps({"db": SPEC["db"], "peak": SPEC["peak_db"],
                    "noise": SPEC["noise_db"], "center": STATE["center_khz"], "span": SPAN_KHZ}))
        elif u.path == "/state":
            with _lock:
                self._send(json.dumps({**{k: STATE[k] for k in
                    ("center_khz", "band", "mode", "ifgr", "running", "lock", "err")},
                    "decode": dict(DECODE), "bands": BANDS, "modes": MODES,
                    "transcript": list(TRANSCRIPT)[-14:],
                    "smeter": round(max(0, (SPEC["peak_db"] - SPEC["noise_db"])), 1)}))
        elif u.path == "/set":
            if "band" in q and q["band"][0] in BANDS:
                STATE["band"] = q["band"][0]; STATE["center_khz"] = float(BANDS[q["band"][0]])
            if "center" in q:
                try: STATE["center_khz"] = round(float(q["center"][0]), 2)
                except ValueError: pass
            if "mode" in q and q["mode"][0] in MODES:
                STATE["mode"] = q["mode"][0]
            if "running" in q:
                STATE["running"] = q["running"][0] == "1"
            self._send(json.dumps({"ok": True}))
        elif u.path == "/autotune":
            with _lock:
                db = np.array(SPEC["db"])
            if len(db):
                off = (int(np.argmax(db)) - len(db) / 2) * (FS / len(db))
                STATE["center_khz"] = round(STATE["center_khz"] + off / 1e3, 2)
            self._send(json.dumps({"ok": True, "center": STATE["center_khz"]}))
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
    SDRWorker().start()
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>hamTuna</title><style>
:root{--bg:#000;--panel:#080c11;--ink:#d6e6f2;--mut:#5f7893;--acc:#2ee6c8;--acc2:#ff5d73;--hair:#141f2b;--good:#3ad17a;--warn:#f0b23a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:ui-monospace,Consolas,monospace;overflow:hidden}
.top{display:flex;align-items:center;gap:14px;padding:8px 14px;border-bottom:1px solid var(--hair);background:var(--panel)}
.logo{font-weight:700;letter-spacing:.06em;color:var(--acc);font-size:18px}.logo b{color:var(--acc2)}
.freq{font-size:30px;font-weight:700;letter-spacing:.04em;color:#fff;text-shadow:0 0 14px rgba(46,230,200,.4)}
.freq small{font-size:13px;color:var(--mut)}.sub{color:var(--mut);font-size:12px}
.wrap{display:grid;grid-template-columns:1fr 320px;height:calc(100vh - 52px)}
.left{display:flex;flex-direction:column;min-width:0}
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
.xscript{background:#000;border:1px solid var(--hair);border-radius:8px;padding:10px;height:150px;overflow-y:auto;font-size:13px;line-height:1.6;display:flex;flex-direction:column-reverse}
.xline{color:var(--acc)}.xline .t{color:var(--mut);font-size:10px;margin-right:6px}
.stat{display:flex;justify-content:space-between;font-size:12px;color:var(--mut);padding:2px 0}.stat b{color:var(--ink)}
.autob{background:var(--acc);color:#04110e;font-weight:700;width:100%;padding:10px;font-size:13px}
.listen{width:100%;padding:9px;font-size:13px;font-weight:700}.listen.on{background:var(--acc2);color:#1a0409;border-color:var(--acc2)}
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
  <div class=left><canvas id=spec></canvas><canvas id=wf></canvas></div>
  <div class=side>
    <div><div class=lbl>Band</div><div class=row id=bands></div></div>
    <div><div class=lbl>Mode</div><div class=row id=modes></div></div>
    <button class=autob onclick=autotune()>&#9673; AUTO-TUNE to strongest</button>
    <button class=listen id=listenb onclick=togListen()>&#9654; LISTEN (live audio)</button>
    <audio id=au></audio>
    <div class=dial>
      <div class=lbl>Truth Dial &mdash; decode confidence</div>
      <div class=big id=conf>0%</div><div class=gauge><div class=gfill id=confbar style=width:0%></div></div>
      <div class=stat><span>WPM</span><b id=wpm>&mdash;</b></div>
      <div class=stat><span>char quality</span><b id=q>&mdash;</b></div>
      <div class=stat><span>S-meter</span><b id=sm>&mdash;</b></div>
      <div class=smeter><div class=sfill id=smbar style=width:0%></div></div>
    </div>
    <div><div class=lbl id=declbl>Live Morse transcript</div><div class=xscript id=xscript></div></div>
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
async function refresh(){
  ST=await api('/state');
  $('freq').innerHTML=ST.center_khz.toFixed(2)+'<small> kHz</small>';
  $('bandlbl').textContent=ST.band+' · '+ST.mode;
  const lk=$('lock');lk.textContent=ST.lock;lk.className='chip '+ST.lock;
  if(!$('bands').dataset.f){$('bands').dataset.f=1;
    for(const b in ST.bands){const e=document.createElement('button');e.textContent=b;e.onclick=()=>set('band='+b);e.dataset.b=b;$('bands').appendChild(e);}
    ST.modes.forEach(m=>{const e=document.createElement('button');e.className='mode';e.textContent=m;e.onclick=()=>set('mode='+m);e.dataset.m=m;$('modes').appendChild(e);});}
  [...$('bands').children].forEach(e=>e.classList.toggle('on',e.dataset.b===ST.band));
  [...$('modes').children].forEach(e=>e.classList.toggle('on',e.dataset.m===ST.mode));
  const d=ST.decode||{};
  $('conf').textContent=Math.round((d.conf||0)*100)+'%';$('confbar').style.width=Math.round((d.conf||0)*100)+'%';
  $('wpm').textContent=d.wpm?d.wpm.toFixed(1):'—';$('q').textContent=d.q?d.q.toFixed(2):'—';
  $('sm').textContent=(ST.smeter||0).toFixed(0)+' dB';$('smbar').style.width=Math.min(100,(ST.smeter||0)*2.2)+'%';
  $('declbl').textContent=ST.mode==='CW'?'Live Morse transcript':ST.mode+' decode';
  const xs=$('xscript');
  if(ST.mode==='CW'){const t=(ST.transcript||[]).slice().reverse();
    xs.innerHTML=t.length?t.map(l=>`<div class=xline><span class=t>${l.ts}</span>${l.text}</div>`).join(''):'<div class=sub>…listening for CW…</div>';}
  else xs.innerHTML='<div class=sub>'+ST.mode+' decode coming soon — spectrum + audio live</div>';
  $('hint').textContent=d.hint||'';
}
async function set(kv){await api('/set?'+kv);refresh();}
async function autotune(){await api('/autotune');refresh();}
// click a canvas -> snap to the local peak near the click (point-and-click nav)
function snap(e,c){const r=c.getBoundingClientRect();const fx=(e.clientX-r.left)/r.width;
  if(!DB.length||!ST.center_khz)return;
  let i0=Math.floor(fx*DB.length),lo=Math.max(0,i0-12),hi=Math.min(DB.length,i0+12),bi=i0,bv=-1e9;
  for(let i=lo;i<hi;i++)if(DB[i]>bv){bv=DB[i];bi=i;}
  const span=ST.span||250,f=ST.center_khz-span/2+(bi/DB.length)*span;set('center='+f.toFixed(2));}
spec.onclick=e=>snap(e,spec);wf.onclick=e=>snap(e,wf);
let listening=false;
function togListen(){const a=$('au');listening=!listening;$('listenb').classList.toggle('on',listening);
  if(listening){a.src='/cw_audio.wav?'+Date.now();a.play().catch(()=>{});$('listenb').innerHTML='&#9632; STOP audio';}
  else{a.pause();a.removeAttribute('src');a.load();$('listenb').innerHTML='&#9654; LISTEN (live audio)';}}
// OLED colormap: weak -> pure black, strong -> cyan -> white-hot
function oled(t){t=Math.max(0,Math.min(1,t));const g2=Math.pow(t,1.4);
  const r=255*Math.pow(Math.max(0,(g2-0.5)*2),1.3),g=255*Math.min(1,g2*1.85),b=255*Math.min(1,g2*1.6);
  return[r|0,g|0,b|0];}
async function draw(){
  let s;try{s=await api('/spectrum');}catch(e){setTimeout(draw,300);return;}
  const db=s.db;DB=db;if(!db.length){setTimeout(draw,150);return;}
  const w=spec.width,h=spec.height;sx.clearRect(0,0,w,h);
  sx.strokeStyle='#0c1a22';for(let i=0;i<=4;i++){const y=h*i/4;sx.beginPath();sx.moveTo(0,y);sx.lineTo(w,y);sx.stroke();}
  const lo=s.noise-6,hi=s.peak+6,rng=Math.max(6,hi-lo);
  sx.strokeStyle='#2ee6c8';sx.lineWidth=1.4;sx.shadowColor='#2ee6c8';sx.shadowBlur=6;sx.beginPath();
  for(let i=0;i<db.length;i++){const x=i/db.length*w,y=h-(db[i]-lo)/rng*h;i?sx.lineTo(x,y):sx.moveTo(x,y);}sx.stroke();sx.shadowBlur=0;
  sx.strokeStyle='rgba(255,93,115,.5)';sx.beginPath();sx.moveTo(w/2,0);sx.lineTo(w/2,h);sx.stroke();
  const cwd=wf.width,ch=wf.height;wx.putImageData(wx.getImageData(0,0,cwd,ch),0,1);
  const row=wx.createImageData(cwd,1);
  for(let x=0;x<cwd;x++){const i=Math.floor(x/cwd*db.length);let v=(db[i]-lo)/rng;const c=oled(v);
    row.data[x*4]=c[0];row.data[x*4+1]=c[1];row.data[x*4+2]=c[2];row.data[x*4+3]=255;}
  wx.putImageData(row,0,0);
  // center marker on waterfall
  wx.fillStyle='rgba(255,93,115,.5)';wx.fillRect(cwd/2,0,1,2);
  setTimeout(draw,140);
}
refresh();setInterval(refresh,1500);draw();
</script></body></html>"""

if __name__ == "__main__":
    main()
