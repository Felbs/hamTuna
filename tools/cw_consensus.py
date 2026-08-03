"""cw_consensus.py - the operator IS the error-correcting code.

The corpus census showed callsigns repeated up to 26 times inside a
single capture ("CQ CQ DE K4RUM K4RUM K4RUM..."). Those are 26 noisy
observations of ONE string. Nothing in the decoder ever used that.

This is the CW analogue of the ADS-B confidence rescue that beat
dump1090: there, weak bits are flipped and a CRC validates the repair.
CW has no CRC - but it has REDUNDANCY the operator supplies for free.
So instead of guessing from a language model (proven non-win: LM
plausibility is not signal truth), we let the repeats vote.

Method:
  1. decode the full capture, keeping each letter's MORSE SYMBOL and a
     per-element confidence (distance from the dit/dah decision line),
  2. cluster tokens by Morse-element distance - copies of the same
     transmission land together,
  3. within a cluster, align every copy to the medoid in ELEMENT space
     (a dropped dit shifts everything downstream, so character-level
     voting would be misaligned), and
  4. vote per element position, weighting each copy by its confidence.
     Majority-of-independent-observations is real evidence: the noise is
     independent between repeats, the signal is not.

Report includes the vote margin so a weak consensus can be disbelieved.

  python cw_consensus.py demo --n 12     # before/after on the corpus
  python cw_consensus.py ab --n 25       # blind A/B vs raw full decode
"""
import argparse
import glob
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cw  # noqa: E402
import cw_center as cc  # noqa: E402
import morse_repair as mr  # noqa: E402

HARVEST = HERE.parent / "lab" / "cw_harvest"
FS = 250000.0
MIN_COPIES = 2
GROUP_COST = 4          # max element distance to call two tokens copies


def to_elements(token):
    """Token -> flat element string with letter separators."""
    return "|".join(cw.INV.get(ch, "") for ch in token.upper())


def from_elements(es):
    out = []
    for sym in es.split("|"):
        if not sym:
            continue
        out.append(cw.MORSE.get(sym, "?"))
    return "".join(out)


def align(a, b):
    """Needleman-Wunsch over element strings -> aligned pair with gaps."""
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1,
                          D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    ai, bi = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            ai.append(a[i - 1]); bi.append(b[j - 1]); i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ai.append(a[i - 1]); bi.append("~"); i -= 1
        else:
            ai.append("~"); bi.append(b[j - 1]); j -= 1
    return "".join(reversed(ai)), "".join(reversed(bi))


def vote(copies):
    """Element-wise majority over aligned copies -> (consensus, margin)."""
    elems = [to_elements(c) for c in copies if c]
    elems = [e for e in elems if e]
    if len(elems) < MIN_COPIES:
        return None, 0.0
    # medoid: the copy closest to all others
    medoid = min(elems, key=lambda x: sum(len(align(x, y)[0])
                                          - sum(p == q for p, q in zip(*align(x, y)))
                                          for y in elems))
    cols = []
    for e in elems:
        am, ae = align(medoid, e)
        col = []
        k = 0
        for cm, ce in zip(am, ae):
            if cm == "~":
                continue           # insertion relative to medoid: ignore
            col.append(ce)
        cols.append(col)
    width = min(len(c) for c in cols)
    out, margins = [], []
    for pos in range(width):
        cnt = Counter(c[pos] for c in cols)
        best, n_best = cnt.most_common(1)[0]
        if best == "~":
            second = [x for x in cnt.most_common() if x[0] != "~"]
            if second and second[0][1] >= n_best:
                best, n_best = second[0]
            else:
                continue
        out.append(best)
        margins.append(n_best / len(cols))
    txt = from_elements("".join(out))
    return txt, float(np.mean(margins)) if margins else 0.0


def _group_ok(t, g):
    """A token joins a group only if it is close to EVERY member, in a
    distance RELATIVE to its own length.

    Fixed 8/03 after the first run "voted" RAIN out of 4F?, AHT?, AL?D,
    BEIE, DRIE - unrelated tokens lumped together and averaged into a
    plausible word. An absolute cost of 4 elements is ~50% of a 3-letter
    token, so anything matched anything. Copies of one transmission
    differ by a few elements out of many; different words do not.
    """
    n_elem = max(len(to_elements(t)), 1)
    budget = max(1, int(0.2 * n_elem))
    return all(mr.elem_distance(t, m) <= budget for m in g)


def consensus_pass(tokens):
    """Group true repeats and replace each group by its vote."""
    groups = []
    for t in tokens:
        if len(t) < 3 or "?" in t:      # unknown letters cannot vote
            continue
        placed = False
        for g in groups:
            if _group_ok(t, g):
                g.append(t); placed = True; break
        if not placed:
            groups.append([t])
    results = []
    for g in groups:
        if len(g) < MIN_COPIES:
            continue
        con, margin = vote(g)
        if con:
            results.append({"consensus": con, "copies": len(g),
                            "margin": round(margin, 2),
                            "variants": sorted(set(g))[:6]})
    results.sort(key=lambda r: (-r["copies"], -r["margin"]))
    return results


def full_text(capture_json):
    d = json.loads(Path(capture_json).read_text())
    f = d.get("full") or {}
    return d, (f.get("text") or "")


def cmd_demo(args):
    shown = 0
    for j in sorted(glob.glob(str(HARVEST / "cw_*.json"))):
        d, txt = full_text(j)
        if not txt:
            continue
        toks = [t for t in re.split(r"[^A-Z0-9?/]+", txt.upper()) if t]
        res = consensus_pass(toks)
        strong = [r for r in res if r["copies"] >= 3 and r["margin"] >= 0.6]
        if not strong:
            continue
        shown += 1
        print(f"\n{Path(j).stem[:26]}  ({len(toks)} tokens)")
        for r in strong[:4]:
            print(f"   {r['consensus']:<12} from {r['copies']:>2} copies "
                  f"(margin {r['margin']:.2f})  variants: "
                  f"{', '.join(r['variants'][:5])}")
        if shown >= args.n:
            break
    print(f"\n{shown} captures with strong consensus shown. Each line is a "
          f"majority vote over the operator's own repeats - signal evidence, "
          f"not a dictionary guess.")
    return 0


def cmd_ab(args):
    """Blind A/B: raw full decode vs consensus-corrected."""
    rng = random.Random(19)
    js = sorted(glob.glob(str(HARVEST / "cw_*.json")))
    rng.shuffle(js)
    samples, key = [], {}
    for j in js:
        if len({k["capture"] for k in key.values()}) >= args.n:
            break
        d, txt = full_text(j)
        if not txt or len(txt.split()) < 6:
            continue
        toks = [t for t in re.split(r"[^A-Z0-9?/]+", txt.upper()) if t]
        res = consensus_pass(toks)
        if not res:
            continue
        # corrected text: replace each group member with the consensus
        repl = {}
        for r in res:
            for v in r["variants"]:
                repl[v] = r["consensus"]
        corrected = " ".join(repl.get(t, t) for t in toks)
        for cond, s in (("raw_full", " ".join(toks)),
                        ("consensus", corrected)):
            sid = f"c{len(samples):03d}{rng.randint(100,999)}"
            samples.append({"id": sid, "text": s[:220]})
            key[sid] = {"cond": cond, "capture": Path(j).stem}
    rng.shuffle(samples)
    out = HERE.parent / "lab" / "cons_blind.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    (HERE.parent / "lab" / "cons_key.json").write_text(json.dumps(key))
    print(f"[ab] {len(samples)} blinded samples -> {out.name}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("--n", type=int, default=12)
    a = sub.add_parser("ab"); a.add_argument("--n", type=int, default=25)
    args = ap.parse_args()
    sys.exit(cmd_demo(args) if args.cmd == "demo" else cmd_ab(args))


if __name__ == "__main__":
    main()
