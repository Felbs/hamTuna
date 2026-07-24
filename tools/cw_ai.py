#!/usr/bin/env python3
"""cw_ai.py - neural Morse decoder (CNN + BiLSTM + CTC). hamTuna's flagship.

The end-to-end learned decoder: it never segments or thresholds - it reads the
whole envelope and outputs text, learning timing, fists, noise and fading from
data. Architecture (after AG1LE's CER-1.5% design, but 1-D on the tone envelope
so it trains on a CPU): envelope@125Hz -> 1-D CNN -> BiLSTM -> CTC over the CW
alphabet. Trained on UNLIMITED synthetic labeled clips (cw_synth) generated on the
fly - no disk, perfect labels - and validated on the real captures.

  python cw_ai.py train --steps 4000     # train (saves lab/morse_ai.pt)
  python cw_ai.py test                    # CER on held-out synthetic + real corpus
  python cw_ai.py decode <file.cs16>      # decode a real capture
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cw
import cw_synth

FS = 250_000.0                     # SDR sample rate of the captures
AUD_AI = 125.0                     # envelope rate fed to the net (AG1LE used 125 Hz)
CHARS = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/?"    # index 0 reserved for CTC blank
BLANK = 0
C2I = {c: i + 1 for i, c in enumerate(CHARS)}        # class 0 = blank, 1.. = chars
I2C = {i + 1: c for i, c in enumerate(CHARS)}
N_CLASSES = len(CHARS) + 1
MODEL_PATH = HERE.parent / "lab" / "morse_ai.pt"
DOWN = 4                            # net time-downsampling factor (two MaxPool1d /2)


def env_to_feat(env, fs):
    """8 kHz magnitude envelope -> normalized 125 Hz feature vector."""
    step = max(1, int(round(fs / AUD_AI)))
    f = env[::step].astype(np.float32)
    # per-clip robust normalize (scale-invariant, like the decision-directed SNR idea)
    med = np.median(f); mad = np.median(np.abs(f - med)) + 1e-6
    return (f - med) / (5 * mad)


def encode_label(text):
    return [C2I[c] for c in text.upper() if c in C2I]


def _build_model():
    import torch.nn as nn

    class MorseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 48, 7, padding=3), nn.BatchNorm1d(48), nn.ReLU(),
                nn.MaxPool1d(2),                                    # /2
                nn.Conv1d(48, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(),
                nn.MaxPool1d(2),                                    # /4
                nn.Conv1d(96, 96, 3, padding=1), nn.BatchNorm1d(96), nn.ReLU(),
                nn.Conv1d(96, 96, 3, padding=1), nn.BatchNorm1d(96), nn.ReLU(),
            )
            self.lstm = nn.LSTM(96, 128, num_layers=2, bidirectional=True,
                                batch_first=True, dropout=0.1)
            self.fc = nn.Linear(256, N_CLASSES)

        def forward(self, x):                # x: (B, 1, T)
            c = self.conv(x).transpose(1, 2)     # (B, T/4, 96)
            o, _ = self.lstm(c)                  # (B, T/4, 256)
            return self.fc(o).log_softmax(-1)    # (B, T/4, classes)

    return MorseNet()


def _real_noise_pool(max_files=30):
    """Harvest REAL band-noise texture (colored/impulsive, not gaussian) from the
    trip-wire captures' key-up gaps, at 8 kHz, normalized. Injecting this into
    synthetic training closes the sim-to-real gap (agent-A's documented fix).
    Returns a 1-D float32 pool, or None if no captures yet."""
    import glob
    cap = HERE.parent / "lab" / "cw_harvest"
    files = sorted(glob.glob(str(cap / "*.cs16")))[:max_files]
    if not files:
        return None
    chunks = []
    for f in files:
        try:
            iq = _load_iq(f)
            off = cw.find_offset(iq, FS, 3000)
            env, _ = cw.envelope(iq, FS, off)                 # 8 kHz
            med = np.median(env); mad = np.median(np.abs(env - med)) + 1e-6
            z = ((env - med) / (5 * mad)).astype(np.float32)
            w = int(0.4 * 8000)                               # 0.4 s windows
            for b in range(0, len(z) - w, w):
                seg = z[b:b + w]
                if np.median(seg) < np.percentile(z, 55):     # keep gap/noise-dominated windows
                    chunks.append(seg)
        except Exception:
            continue
    return np.concatenate(chunks) if chunks else None


EXACT_IQ = False                   # EXP-4 tested: exact IQ->cw.envelope is 30x slower
#                                    data-gen (6.2s/batch) -> impractical; EXP-3 retained


def _synth_clip(rng, noise_pool=None):
    """One labeled clip -> (feat, label_ids, text). EXACT_IQ path synthesizes IQ and
    runs the SAME cw.envelope pipeline as real captures (train==test processing)."""
    text = cw_synth.random_text(rng)
    use_real = noise_pool is not None and rng.random() < 0.5
    if EXACT_IQ:
        iq, fsq, f0 = cw_synth.render_iq(
            text, wpm=int(rng.integers(12, 34)), fs_iq=24000.0, f0=700.0,
            jitter=float(rng.uniform(0.05, 0.22)), weight=float(rng.uniform(0.9, 1.35)),
            noise=float(rng.uniform(0.02, 0.08) if use_real else rng.uniform(0.05, 0.45)),
            fade=float(rng.choice([0.0, 0.0, 0.4, 0.6, 0.75])),
            fade_hz=float(rng.uniform(0.2, 1.5)), rise_ms=float(rng.uniform(2, 8)),
            seed=int(rng.integers(1, 1_000_000)))
        env, fs = cw.envelope(iq, fsq, f0)                    # identical to real capture path
        if use_real and len(noise_pool) > len(env):
            s = int(rng.integers(0, len(noise_pool) - len(env)))
            env = np.abs(env + rng.uniform(0.4, 1.3) * noise_pool[s:s + len(env)])
        return env_to_feat(env, fs), encode_label(text), text
    env, fs = cw_synth.render(
        text, wpm=int(rng.integers(12, 34)), jitter=float(rng.uniform(0.05, 0.22)),
        weight=float(rng.uniform(0.9, 1.35)),
        noise=float(rng.uniform(0.02, 0.08) if use_real else rng.uniform(0.08, 0.35)),
        fade=float(rng.choice([0.0, 0.0, 0.4, 0.6, 0.75])),
        fade_hz=float(rng.uniform(0.2, 1.5)),
        rise_ms=float(rng.uniform(2, 8)), qrn=float(rng.choice([0, 0, 2, 6])),
        seed=int(rng.integers(1, 1_000_000)))
    if use_real and len(noise_pool) > len(env):
        s = int(rng.integers(0, len(noise_pool) - len(env)))
        env = np.abs(env + rng.uniform(0.4, 1.3) * noise_pool[s:s + len(env)])
    return env_to_feat(env, fs), encode_label(text), text


def _batch(rng, n, torch, noise_pool=None):
    feats, labels, ilens, tlens = [], [], [], []
    for _ in range(n):
        f, lab, _ = _synth_clip(rng, noise_pool)
        if len(lab) < 1 or len(f) < DOWN * (len(lab) + 2):
            continue                     # CTC needs input_len >= label_len
        feats.append(f); labels.append(lab)
        ilens.append(len(f) // DOWN); tlens.append(len(lab))
    if not feats:
        return None
    T = max(len(f) for f in feats)
    X = np.zeros((len(feats), 1, T), np.float32)
    for i, f in enumerate(feats):
        X[i, 0, :len(f)] = f
    y = torch.tensor([c for lab in labels for c in lab], dtype=torch.long)
    return (torch.from_numpy(X), y,
            torch.tensor(ilens, dtype=torch.long), torch.tensor(tlens, dtype=torch.long))


def ctc_greedy(logp):
    """logp: (T, classes) numpy -> text (collapse repeats, drop blanks)."""
    idx = logp.argmax(-1)
    out, prev = [], -1
    for i in idx:
        if i != prev and i != BLANK:
            out.append(I2C.get(int(i), ""))
        prev = i
    return "".join(out)


def decode_feat(model, feat, torch):
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        x = torch.from_numpy(feat[None, None, :].astype(np.float32)).to(dev)
        lp = model(x)[0].cpu().numpy()
    return ctc_greedy(lp)


def _producer(q, stop, batch, seed, torch, noise_pool=None):
    """Background batch generator: overlaps CPU synth with GPU compute so the 4090
    isn't starved. Several of these run in parallel across the Threadripper cores."""
    rng = np.random.default_rng(seed)
    while not stop.is_set():
        b = _batch(rng, batch, torch, noise_pool)
        if b is not None:
            try:
                q.put(b, timeout=1.0)
            except Exception:
                pass


def train(steps=20000, batch=48, lr=2e-3, seed=1, workers=6, resume=True):
    import queue
    import threading
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {dev} "
          f"({torch.cuda.get_device_name(0) if dev.type=='cuda' else 'CPU'})  "
          f"batch={batch} workers={workers} steps={steps}", flush=True)
    model = _build_model().to(dev)
    if resume and MODEL_PATH.exists():
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
            print(f"resumed from {MODEL_PATH}", flush=True)
        except Exception:
            pass
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # long, gently-decaying schedule with a floor so short chunks don't kill the LR
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 40000), eta_min=3e-4)
    ctc = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
    # parallel prefetch producers -> queue (infinite fresh data, GPU-fed)
    noise_pool = _real_noise_pool()
    print(f"real-noise pool: {'%d samples (%.0fs)' % (len(noise_pool), len(noise_pool)/8000) if noise_pool is not None else 'none (synth-only)'}", flush=True)
    q, stop = queue.Queue(maxsize=16), threading.Event()
    threads = [threading.Thread(target=_producer, args=(q, stop, batch, seed + 100 + i, torch, noise_pool),
                                daemon=True) for i in range(workers)]
    for t in threads:
        t.start()
    real_val = _load_real_val()
    print(f"real validation set: {len(real_val)} hand-labeled capture(s) "
          f"-> checkpoint on {'REAL' if real_val else 'synth'} CER", flush=True)
    model.train()
    run, best = 0.0, 9.9
    try:
        for step in range(1, steps + 1):
            X, y, il, tl = q.get()
            lp = model(X.to(dev)).transpose(0, 1)        # (T, B, C)
            il = torch.clamp(il, max=lp.shape[0])
            loss = ctc(lp, y.to(dev), il, tl)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            run += float(loss.detach())
            if step % 200 == 0:
                vc = _val_cer(model, torch, n=60, seed=999)
                rc = _real_cer(model, real_val, torch)
                # checkpoint on REAL CER (the metric that matters); synth as tiebreak
                metric = rc if rc is not None else vc
                print(f"  step {step:6d}/{steps}  loss={run/200:.3f}  "
                      f"synthCER={vc:.3f}  realCER={rc if rc is None else round(rc,3)}  "
                      f"lr={sched.get_last_lr()[0]:.1e}", flush=True)
                run = 0.0
                MODEL_PATH.parent.mkdir(exist_ok=True)
                if metric <= best:
                    best = metric
                    torch.save(model.state_dict(), MODEL_PATH)
                    print(f"    * checkpoint (best {'real' if rc is not None else 'synth'}CER={best:.3f})", flush=True)
    finally:
        stop.set()
    MODEL_PATH.parent.mkdir(exist_ok=True)
    if best >= 9.9:
        torch.save(model.state_dict(), MODEL_PATH)
    print(f"done. best CER={best:.3f}  saved {MODEL_PATH}", flush=True)
    return model


def load_model():
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model().to(dev)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    model.eval()
    return model, torch


def _load_real_val():
    """Load hand-labeled real captures (feat, text) for REAL validation - the
    metric that actually matters (synth val_CER overfits, doesn't predict real)."""
    import json
    lf = HERE.parent / "lab" / "real_labels.json"
    cap = HERE.parent / "lab" / "cw_harvest"
    corp = HERE.parent / "lab" / "cw_corpus"
    if not lf.exists():
        return []
    out = []
    for fn, text in json.loads(lf.read_text()).items():
        p = (corp / fn) if (corp / fn).exists() else (cap / fn)
        if not p.exists():
            continue
        try:
            iq = _load_iq(str(p)); off = cw.find_offset(iq, FS, 50000)
            env, aud = cw.envelope(iq, FS, off)
            out.append((env_to_feat(env, aud), text.upper()))
        except Exception:
            continue
    return out


def _real_cer(model, real_val, torch):
    if not real_val:
        return None
    import cw_lm
    tot_e, tot_n = 0, 0
    for feat, text in real_val:
        pred = cw_lm.rescore(decode_feat(model, feat, torch))
        tot_e += cw_synth._cer(pred, text) * max(1, len(text.replace(" ", "")))
        tot_n += max(1, len(text.replace(" ", "")))
    model.train()
    return tot_e / max(tot_n, 1)


def _val_cer(model, torch, n=40, seed=999):
    rng = np.random.default_rng(seed)
    tot_e, tot_n = 0, 0
    for _ in range(n):
        f, lab, text = _synth_clip(rng)
        pred = decode_feat(model, f, torch)
        tot_e += cw_synth._cer(pred, text) * max(1, len(text.replace(" ", "")))
        tot_n += max(1, len(text.replace(" ", "")))
    model.train()
    return tot_e / max(tot_n, 1)


def _load_iq(f):
    raw = np.fromfile(f, np.int16)
    return ((raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)) / 32768.0).astype(np.complex64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=48)
    sub.add_parser("test")
    d = sub.add_parser("decode"); d.add_argument("file")
    a = ap.parse_args()
    if a.cmd == "train":
        train(steps=a.steps, batch=a.batch)
    elif a.cmd == "test":
        model, torch = load_model()
        print(f"held-out synthetic CER: {_val_cer(model, torch, n=100, seed=555):.3f}")
        import glob
        import cw_lm
        for f in sorted(glob.glob(str(HERE.parent / "lab" / "cw_corpus" / "*.cs16"))):
            iq = _load_iq(f); off = cw.find_offset(iq, 250000.0, 50000)
            env, aud = cw.envelope(iq, 250000.0, off)
            raw = decode_feat(model, env_to_feat(env, aud), torch)
            print(f"  {Path(f).name[:26]:26} AI: {cw_lm.rescore(raw)[:52]!r}")
    elif a.cmd == "decode":
        model, torch = load_model()
        import cw_lm
        iq = _load_iq(a.file); off = cw.find_offset(iq, 250000.0, 50000)
        env, aud = cw.envelope(iq, 250000.0, off)
        print(cw_lm.rescore(decode_feat(model, env_to_feat(env, aud), torch)))


if __name__ == "__main__":
    main()
