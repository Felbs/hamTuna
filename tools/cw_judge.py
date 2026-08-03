"""cw_judge.py - blind LLM adjudication of CW decodes.

The callsign "ground truth" turned out circular (labels produced by the
decoder, then DB-checked - 8/03). A reader who was not involved in
producing the text is an INDEPENDENT oracle, and an LLM can read a lot
of it. The catch is bias: if the judge knows which condition produced a
sample, the judgement is worthless.

So this harness enforces blinding:

  blind   - decode each sampled capture under every condition (aims and
            decoders), shuffle all samples together, assign opaque ids,
            and write ONLY the text to lab/judge_blind.jsonl. The key
            (id -> condition) is written separately to lab/judge_key.json
            which the judge must not read.
  score   - ingest lab/judge_scores.json ({id: 0..3}), unblind, and
            report per-condition means with a win/loss tally.

Scoring rubric handed to the judge (0-3):
  0  gibberish: no CW structure, random letters
  1  fragments: isolated real tokens (K, E, T runs) but nothing readable
  2  partial QSO: recognisable prosigns/callsigns/exchange amid errors
  3  readable: a human could follow the exchange

  python cw_judge.py blind --n 40
  (judge reads lab/judge_blind.jsonl, writes lab/judge_scores.json)
  python cw_judge.py score
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cw  # noqa: E402
import cw_center as cc  # noqa: E402

LAB = HERE.parent / "lab"
HARVEST = LAB / "cw_harvest"
BLIND = LAB / "judge_blind.jsonl"
KEY = LAB / "judge_key.json"
SCORES = LAB / "judge_scores.json"
FS = 250000.0


def decode_at(iq, hz):
    env, aud = cw.envelope2(iq, FS, float(hz), aud=8000, bw_hz=150)
    t = cw.decode_env_auto(env, aud)
    if isinstance(t, (list, tuple)):
        t = " ".join(str(x) for x in t)
    return (t or "").split("{")[0].strip()


def cmd_blind(args):
    rng = random.Random(args.seed)
    js = sorted(glob.glob(str(HARVEST / "cw_*.json")))
    rng.shuffle(js)
    samples, key = [], {}
    for j in js:
        if len(samples) >= args.n * 2:
            break
        d = json.loads(Path(j).read_text())
        c = d.get("center") or {}
        r = d.get("relabel") or {}
        p = HARVEST / d.get("iq_file", "")
        if not p.is_file() or c.get("hz") is None or r.get("off_hz") is None:
            continue
        if not c.get("is_cw"):
            continue                      # judge the CW-bearing captures
        try:
            iq = cc._load(p, n_max=int(8 * FS))
        except Exception:
            continue
        for cond, hz in (("center", c["hz"]), ("old_aim", r["off_hz"])):
            txt = decode_at(iq, hz)
            if not txt:
                continue
            sid = f"s{len(samples):03d}{rng.randint(100, 999)}"
            samples.append({"id": sid, "text": txt[:220]})
            key[sid] = {"cond": cond, "capture": Path(j).stem,
                        "hz": round(float(hz), 1)}
    rng.shuffle(samples)
    with open(BLIND, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    KEY.write_text(json.dumps(key, indent=1))
    print(f"[judge] wrote {len(samples)} blinded samples -> {BLIND.name}")
    print(f"[judge] key hidden in {KEY.name} - DO NOT READ IT before scoring")
    print("[judge] rubric: 0 gibberish / 1 fragments / 2 partial QSO / "
          "3 readable")
    return 0


def cmd_score(args):
    if not SCORES.is_file():
        print(f"no {SCORES.name} - the judge must write {{id: 0..3}} first")
        return 1
    scores = json.loads(SCORES.read_text())
    key = json.loads(KEY.read_text())
    by = {}
    for sid, sc in scores.items():
        k = key.get(sid)
        if not k:
            continue
        by.setdefault(k["cond"], []).append((float(sc), k["capture"]))
    print(f"{'condition':<12}{'n':>5}{'mean':>8}{'>=2':>6}{'=3':>5}")
    for cond, rows in sorted(by.items()):
        v = np.array([r[0] for r in rows])
        print(f"{cond:<12}{len(v):>5}{v.mean():>8.2f}"
              f"{int((v >= 2).sum()):>6}{int((v == 3).sum()):>5}")
    # paired comparison on captures judged under both conditions
    per = {}
    for cond, rows in by.items():
        for sc, cap in rows:
            per.setdefault(cap, {})[cond] = sc
    both = {c: v for c, v in per.items() if len(v) == 2}
    if both:
        w = sum(1 for v in both.values() if v.get("center", 0) > v.get("old_aim", 0))
        l = sum(1 for v in both.values() if v.get("center", 0) < v.get("old_aim", 0))
        print(f"\npaired on {len(both)} captures: auto-center wins {w}, "
              f"loses {l}, ties {len(both) - w - l}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("blind")
    b.add_argument("--n", type=int, default=40)
    b.add_argument("--seed", type=int, default=11)
    sub.add_parser("score")
    a = ap.parse_args()
    sys.exit(cmd_blind(a) if a.cmd == "blind" else cmd_score(a))


if __name__ == "__main__":
    main()
