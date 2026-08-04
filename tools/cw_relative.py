"""cw_relative.py - decode by RELATIVE mark length, the way an eye reads it.

User's observation (8/03): "when I look at it with my human eyes I can
clearly make out shorter and longer lines - the longer being dashes and
the shorter dots." That is a LOCAL, RELATIVE judgement, and it is
strictly more robust than what the classic decoder does: estimate ONE
dit length for the whole capture and compare every mark to it. We
measured ~24% speed drift *within* captures, so a global ruler
misjudges the marks at the far end of a transmission.

This decoder therefore:

  * slices keying with a Schmitt trigger on a smoothed envelope
    (cw_center.key_runs - built because raw slicing chopped noise into
    fake 3-15 ms dits),
  * classifies marks by 2-MEANS OVER A SLIDING WINDOW: within each
    window the marks split into a short cluster and a long cluster, so
    the ruler re-calibrates as the operator speeds up, slows down or
    fades. No absolute threshold anywhere,
  * classifies gaps against the SAME local dit (element / letter /
    word), which is what makes Farnsworth spacing and hesitation
    survivable,
  * reports per-element CONFIDENCE as the normalised distance from the
    cluster boundary - the same idea as the ADS-B confidence plane, and
    the hook for N-best alternatives later.

  python cw_relative.py test                 # synthetic ground truth
  python cw_relative.py ab --n 30            # blind A/B vs the classic
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cw  # noqa: E402
import cw_center as cc  # noqa: E402

HARVEST = HERE.parent / "lab" / "cw_harvest"
FS = 250000.0  # rate-ok: corpus-file read rate, not a capture path
WIN = 24            # marks per sliding window for the local ruler
MIN_WIN = 8


def _split2(vals):
    """2-means on log durations -> (short_center, long_center). The eye's
    'these are shorter, those are longer' made arithmetic."""
    lg = np.log(np.clip(np.asarray(vals, float), 1e-6, None))
    if len(lg) < 2 or lg.max() - lg.min() < 0.12:
        m = float(np.exp(lg.mean()))
        return m, m * 3.0            # single cluster: assume all dits
    c1, c2 = np.percentile(lg, 20), np.percentile(lg, 80)
    for _ in range(30):
        m = np.abs(lg - c1) <= np.abs(lg - c2)
        if not m.any() or m.all():
            break
        c1, c2 = float(lg[m].mean()), float(lg[~m].mean())
    return float(np.exp(min(c1, c2))), float(np.exp(max(c1, c2)))


def decode(env, aud, want_conf=False):
    """Envelope -> text, using local relative timing throughout."""
    runs = cc.key_runs(env, aud)
    if len(runs) < 6:
        return ("", {}) if not want_conf else ("", {}, [])
    marks = [(i, ln) for i, (s, ln) in enumerate(runs) if s and ln > 1]
    if len(marks) < MIN_WIN:
        return ("", {}) if not want_conf else ("", {}, [])
    mark_lens = [m[1] for m in marks]
    # local ruler per mark: 2-means over the surrounding WIN marks
    dits, dahs = [], []
    for k in range(len(marks)):
        a = max(0, k - WIN // 2)
        b = min(len(marks), a + WIN)
        a = max(0, b - WIN)
        d, h = _split2(mark_lens[a:b])
        dits.append(d)
        dahs.append(h)
    sym, out, confs = "", [], []
    mi = 0
    for i, (s, ln) in enumerate(runs):
        if s:
            if ln <= 1:
                continue
            d, h = dits[mi], dahs[mi]
            mid = np.sqrt(max(d, 1e-9) * max(h, 1e-9))   # geometric midline
            sym += "-" if ln > mid else "."
            # confidence: how far from the decision line, in local dits
            confs.append(min(1.0, abs(np.log(max(ln, 1e-9) / mid))
                             / max(np.log(max(h / max(d, 1e-9), 1.2)) / 2, 1e-6)))
            mi += 1
        else:
            if mi == 0 or mi >= len(dits):
                continue
            d = dits[min(mi, len(dits) - 1)]
            # gaps measured in LOCAL dits: <2 element, 2-5 letter, >5 word
            g = ln / max(d, 1e-9)
            if g >= 2.0 and sym:
                out.append(cw.MORSE.get(sym, "?"))
                sym = ""
                if g >= 5.0:
                    out.append(" ")
    if sym:
        out.append(cw.MORSE.get(sym, "?"))
    text = "".join(out)
    info = {"marks": len(marks),
            "dit_ms": round(float(np.median(dits)) / aud * 1000, 1),
            "wpm": round(1200.0 / max(float(np.median(dits)) / aud * 1000, 1e-6)),
            "mean_conf": round(float(np.mean(confs)) if confs else 0.0, 2)}
    return (text, info, confs) if want_conf else (text, info)


def synth(msg, wpm, snr_db, drift=0.0, seed=3, fs=8000.0):
    """Keyed envelope with optional SPEED DRIFT - the thing a global dit
    estimate cannot survive and a local ruler can."""
    rng = np.random.default_rng(seed)
    parts = []
    n = 0
    total = sum(len(cw.INV.get(c, "")) for c in msg) or 1
    for ch in msg.upper():
        if ch == " ":
            parts.append(np.zeros(int(fs * 1.2 / wpm * 4)))
            continue
        for el in cw.INV.get(ch, ""):
            k = 1.2 / wpm * (1.0 + drift * (n / total - 0.5))
            n += 1
            parts.append(np.ones(int(fs * k * (3 if el == "-" else 1))))
            parts.append(np.zeros(int(fs * k)))
        parts.append(np.zeros(int(fs * 1.2 / wpm * 2)))
    env = np.concatenate(parts).astype(np.float32)
    amp = 10 ** (-snr_db / 20.0)
    return np.abs(env + rng.normal(0, amp, len(env))).astype(np.float32), fs


def cer(a, b):
    a, b = a.replace(" ", ""), b.replace(" ", "")
    if not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ca != cb))
        prev = cur
    return prev[len(b)] / len(b)


def cmd_test(args):
    msg = "CQ CQ DE W1AW W1AW K"
    print(f"{'case':<28}{'classic CER':>13}{'relative CER':>14}")
    rows = []
    for wpm, snr, drift, label in ((18, 20, 0.0, "18wpm clean"),
                                   (18, 8, 0.0, "18wpm  8dB"),
                                   (25, 20, 0.0, "25wpm clean"),
                                   (18, 20, 0.5, "18wpm +50% DRIFT"),
                                   (18, 12, 0.5, "18wpm 12dB +DRIFT"),
                                   (30, 20, 0.3, "30wpm +30% drift")):
        env, aud = synth(msg, wpm, snr, drift)
        t_classic, _ = cw.decode_env(env, aud)
        t_rel, _ = decode(env, aud)
        c1, c2 = cer(t_classic, msg), cer(t_rel, msg)
        rows.append((c1, c2))
        print(f"{label:<28}{c1:>13.2f}{c2:>14.2f}")
    a = np.array(rows)
    print(f"\nmean CER: classic {a[:,0].mean():.2f}  relative {a[:,1].mean():.2f}")
    print("(0.00 = perfect copy; drift cases are where a global dit "
          "estimate is expected to fail)")
    return 0


def cmd_ab(args):
    """Decode real captures both ways; write a blind file for judging."""
    import random
    rng = random.Random(7)
    js = sorted(glob.glob(str(HARVEST / "cw_*.json")))
    rng.shuffle(js)
    samples, key = [], {}
    for j in js:
        if len({k["capture"] for k in key.values()}) >= args.n:
            break
        d = json.loads(Path(j).read_text())
        c = d.get("center") or {}
        if not c.get("is_cw") or c.get("hz") is None:
            continue
        p = HARVEST / d.get("iq_file", "")
        if not p.is_file():
            continue
        try:
            iq = cc._load(p, n_max=int(60 * FS))
            env, aud = cw.envelope2(iq, FS, float(c["hz"]), aud=8000,
                                    bw_hz=150)
        except Exception:
            continue
        t_classic = cw.decode_env_auto(env, aud)
        if isinstance(t_classic, (list, tuple)):
            t_classic = " ".join(str(x) for x in t_classic)
        t_classic = (t_classic or "").split("{")[0].strip()
        t_rel, _ = decode(env, aud)
        for cond, txt in (("classic", t_classic), ("relative", t_rel)):
            if not txt.strip():
                continue
            sid = f"r{len(samples):03d}{rng.randint(100,999)}"
            samples.append({"id": sid, "text": txt[:200]})
            key[sid] = {"cond": cond, "capture": Path(j).stem}
    rng.shuffle(samples)
    out = HERE.parent / "lab" / "relab_blind.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    (HERE.parent / "lab" / "relab_key.json").write_text(json.dumps(key))
    print(f"[ab] {len(samples)} blinded samples -> {out.name} "
          f"(key hidden; judge before unblinding)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test")
    a = sub.add_parser("ab")
    a.add_argument("--n", type=int, default=25)
    args = ap.parse_args()
    sys.exit(cmd_test(args) if args.cmd == "test" else cmd_ab(args))


if __name__ == "__main__":
    main()
