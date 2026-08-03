#!/usr/bin/env python3
"""cw_finetune.py - overnight fine-tune of the neural CW decoder on the REAL corpus.

Campaign-1 verdict stands: classic decode_env_auto+LM is production (13/13 real
callsigns), the synth-trained net was 0/13 - a sim-to-real DISTRIBUTION gap. The
honest question this run answers: does training on REAL harvested envelopes (the
big-corpus mission + tonight's best-eye relabel pass) close that gap?

Recipe:
  * REAL pool: every relabeled capture with a usable label (label_conf verified/
    decode), decoded at its best-eye offset - EXCLUDING the callsign-recall
    holdout (original verified_calls + eye>=3.0) so the metric stays honest.
  * 50/50 mixed batches: real clips + domain-randomized synthetic (EXP-3 Rician
    stats, real-noise injection) so ~500 real clips can't be memorized.
  * Warm start from lab/morse_ai.pt; low LR (fine-tune, not re-train).
  * Checkpoint on CALLSIGN RECALL (the metric that exposed Campaign 1's overfit),
    tie-broken by real-label CER -> lab/morse_ai_real.pt. Classic's bar: recall
    printed alongside for the morning verdict.

  python cw_finetune.py --smoke          # 2-min sanity (small pool, 30 steps)
  python cw_finetune.py --until 07:45    # the overnight run
"""
import argparse
import glob
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import cw
import cw_ai
from cw_ai import (AUD_AI, DOWN, FS, MODEL_PATH, _build_model, _callsign_recall,
                   _load_callsign_val, _load_iq, _load_real_val, _real_cer,
                   _real_noise_pool, _synth_clip, encode_label, env_to_feat)

HERE = Path(__file__).resolve().parent
MIN_EYE = [0.0]
SOURCE = ["relabel"]
HARVEST = HERE.parent / "lab" / "cw_harvest"
OUT_PATH = HERE.parent / "lab" / "morse_ai_real.pt"
MAX_SECS = 24.0                      # per-capture clip cap (matches training scale)


def _holdout_names():
    """Files the callsign-recall metric uses - NEVER train on these."""
    names = set()
    for j in glob.glob(str(HARVEST / "*.json")):
        try:
            d = json.loads(Path(j).read_text())
        except Exception:
            continue
        if d.get("verified_calls") and float(d.get("eye", 0)) >= 3.0:
            names.add(d.get("iq_file", ""))
    return names


def _one_real(args):
    jp, holdout = args
    try:
        d = json.loads(Path(jp).read_text())
        rl = d.get("relabel")
        # 8/03 FINDING: the "decode"-tier label is the CLASSIC decoder's
        # own output - training on it DISTILLS ITS ERRORS. Two runs (lr
        # 5e-4 and 1e-4) both showed real-label CER DEGRADING with training
        # (0.42 -> 0.55/0.79) while loss fell: the net was learning the
        # teacher's noise on marginal clips. --min-eye keeps only clips
        # where the teacher is trustworthy (eye is the copy-quality dial).
        # SOURCE (8/03): --source center uses the AUTO-CENTERED aim and its
        # decode. The July `relabel` aim was often on the wrong carrier
        # (that is what sank rounds 1-2), and `center.is_cw` additionally
        # certifies the capture actually contains keying - so this pool is
        # both correctly aimed and CW-verified.
        ctr = d.get("center") or {}
        if SOURCE[0] == "center":
            if not ctr.get("is_cw") or ctr.get("hz") is None:
                return None
            text = ctr.get("text")
            if isinstance(text, (list, tuple)):
                text = " ".join(str(x) for x in text)
            text = (text or "").split("{")[0].strip()
            off_hz = float(ctr["hz"])
            if float(ctr.get("rhythm", 0)) < MIN_EYE[0] / 10.0:
                return None
        else:
            if not rl or rl.get("label_conf") not in ("verified", "decode"):
                return None
            if float(rl.get("eye", 0)) < MIN_EYE[0]:
                return None
            text = rl.get("text", "")
            off_hz = float(rl.get("off_hz", 0.0))
        if d.get("iq_file", "") in holdout:
            return None
        lab = encode_label(text)
        if len(lab) < 6:
            return None
        iq = _load_iq(str(HARVEST / d["iq_file"]))[: int(MAX_SECS * FS)]
        env, aud = cw.envelope(iq, FS, off_hz)
        feat = env_to_feat(env, aud)
        if len(feat) < DOWN * (len(lab) + 2):
            return None
        return feat.astype(np.float32), lab
    except Exception:
        return None


def build_real_pool(limit=None, workers=8):
    js = [j for j in sorted(glob.glob(str(HARVEST / "*.json"))) if "besteye" not in j]
    if limit:
        js = js[:limit]
    holdout = _holdout_names()
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        out = [r for r in ex.map(_one_real, ((j, holdout) for j in js)) if r]
    print(f"[finetune] real pool: {len(out)} clips from {len(js)} caps "
          f"(holdout excluded: {len(holdout)}) in {(time.time()-t0)/60:.1f} min",
          flush=True)
    return out


def run(until="07:45", batch=48, max_steps=60000, lr=5e-4, seed=7, smoke=False):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = build_real_pool(limit=40 if smoke else None)
    if len(pool) < 8:
        print("[finetune] pool too small - aborting"); return
    noise_pool = _real_noise_pool()
    cval, rval = _load_callsign_val(), _load_real_val()
    print(f"[finetune] dev={dev} holdout-val={len(cval)} caps, "
          f"label-val={len(rval)}", flush=True)
    model = _build_model().to(dev)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
        print(f"[finetune] warm-started from {MODEL_PATH.name}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=min(1e-4, lr / 5))
    ctc = torch.nn.CTCLoss(blank=cw_ai.BLANK, zero_infinity=True)
    rng = np.random.default_rng(seed)

    # deadline: HH:MM, tomorrow-aware
    hh, mm = (int(x) for x in until.split(":"))
    now = time.localtime()
    dl = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hh, mm, 0, 0, 0, -1))
    if dl < time.time():
        dl += 86400
    if smoke:
        dl = time.time() + 180
        max_steps = 30

    model.train()
    best, run_loss, step = -1.0, 0.0, 0
    best_score = (-2.0, -9e9)
    while step < max_steps and time.time() < dl:
        step += 1
        clips = [pool[i] for i in rng.integers(0, len(pool), batch // 2)]
        for _ in range(batch - len(clips)):
            f, l, _t = _synth_clip(rng, noise_pool)
            clips.append((f, l))
        keep = [(f, l) for f, l in clips if len(f) >= DOWN * (len(l) + 2) and len(l) > 0]
        if len(keep) < 2:
            continue
        T = max(len(f) for f, _ in keep)
        X = np.zeros((len(keep), 1, T), np.float32)
        for i, (f, _) in enumerate(keep):
            X[i, 0, : len(f)] = f
        y = torch.tensor([c for _, l in keep for c in l], dtype=torch.long)
        in_lens = torch.tensor([len(f) // DOWN for f, _ in keep], dtype=torch.long)
        lab_lens = torch.tensor([len(l) for _, l in keep], dtype=torch.long)
        logp = model(torch.from_numpy(X).to(dev)).transpose(0, 1)  # -> (T', N, C)
        in_lens = torch.clamp(in_lens, max=logp.shape[0])
        loss = ctc(logp, y.to(dev), in_lens, lab_lens)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        run_loss += float(loss)
        if step % (10 if smoke else 300) == 0:
            rec = _callsign_recall(model, cval, torch)
            rcer = _real_cer(model, rval, torch)
            print(f"[finetune] step {step} loss={run_loss/(10 if smoke else 300):.3f} "
                  f"recall={rec} realCER={rcer} lr={sched.get_last_lr()[0]:.1e}",
                  flush=True)
            run_loss = 0.0
            # BUGFIX 8/03: `rec >= best` saved on EVERY eval while recall
            # sat at 0.0, so the surviving artifact was the LAST model, not
            # the best - and with real-label CER climbing that meant saving
            # the worst. Implement the tie-break the docstring promised:
            # rank by (recall, -CER) so a recall tie is decided by CER.
            score = (rec if rec is not None else -1.0, -(rcer if rcer
                                                         is not None else 9e9))
            if score > best_score:
                best_score = score
                best = rec
                torch.save(model.state_dict(), OUT_PATH)
                print(f"[finetune]   checkpoint -> {OUT_PATH.name} (recall {rec}, CER {rcer:.3f})",
                      flush=True)
    print(f"\n[finetune] MORNING REPORT: best holdout callsign-recall={best} "
          f"(classic decoder's bar on this rig: ~1.0 at power-argmax aim; Campaign-1 "
          f"synth-only net: 0.0). Model: {OUT_PATH}. Steps run: {step}.", flush=True)
    print("[finetune] NOT wired into the panel - opt-in eval first "
          "(python cw_ai.py test w/ morse_ai_real.pt, then decide).", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default="07:45")
    ap.add_argument("--source", choices=("relabel", "center"),
                    default="relabel",
                    help="which aim+label to train on; 'center' = the "
                         "auto-centered, CW-verified pool (8/03)")
    ap.add_argument("--min-eye", type=float, default=0.0,
                    help="drop clips whose eye-opening is below this - the "
                         "teacher-label trust gate (8/03 finding)")
    ap.add_argument("--lr", type=float, default=5e-4,
                    help="fine-tune LR (8/03: 5e-4 warm-started too hot - "
                         "loss fell while real-label CER rose 0.47->0.79, "
                         "the synth half pulling the model off real data)")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    MIN_EYE[0] = a.min_eye
    SOURCE[0] = a.source
    run(until=a.until, lr=a.lr, batch=a.batch, max_steps=a.max_steps, smoke=a.smoke)
