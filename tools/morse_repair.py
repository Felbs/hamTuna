"""morse_repair.py - read through the operator's errors, without inventing.

A garbled CW decode is not random text: the errors are ELEMENT errors.
A dropped dit turns I into E, an extra dah turns T into M, a merged gap
turns "K N" into "KN". So the honest way to "guess" what was sent is a
MORSE-DISTANCE search against a constrained lexicon - never a language
model free-associating a plausible QSO.

Two evidence sources, both auditable:

  1. LEXICON REPAIR - map a token to the nearest entry in a closed set
     (prosigns, standard CW abbreviations, contest exchange forms) when
     the Morse-element edit distance is small. Every repair records its
     cost, so a reader can see what was changed and disbelieve it.

  2. REPETITION CONSENSUS - CW operators repeat: "CQ CQ DE W1AW W1AW K".
     Two garbled copies of the same token are INTERNAL evidence; merging
     them element-wise is inference from the signal, not from a prior.
     This is the strong one: it can recover a callsign that is in no
     dictionary.

Explicitly NOT done: filling gaps with likely words, completing partial
sentences, or accepting a repair that no evidence supports. The
campaign already learned that LM-plausibility is not signal-truth
(EXP-8/9 non-win) - and that was on MIS-AIMED audio, so this re-test
runs on the auto-centered corpus with the fabrication risk stated up
front and every repair labelled with its evidence.

  python morse_repair.py demo            # before/after on the corpus
  python morse_repair.py full --n 40     # full-length decode + repair
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cw  # noqa: E402
import cw_center as cc  # noqa: E402

HARVEST = HERE.parent / "lab" / "cw_harvest"
FS = 250000.0

# closed lexicon: what CW operators actually send
PROSIGNS = ["CQ", "DE", "K", "KN", "AR", "SK", "BK", "R", "TU", "73", "88",
            "QRZ", "QSL", "QTH", "QRM", "QRN", "QSB", "QRP", "QSY", "QST",
            "RST", "5NN", "599", "579", "559", "TEST", "POTA", "SOTA",
            "IOTA", "DX", "ES", "OM", "YL", "TNX", "FB", "HW", "PSE", "AGN",
            "UP", "DN", "CFM", "GM", "GA", "GE", "GN", "WX", "ANT", "PWR",
            "RIG", "NAME", "OP", "HR", "NR", "CU", "GL", "VY", "ABT", "WID"]
CALLSIGN = re.compile(r"^[A-Z]{1,2}[0-9][A-Z]{1,4}$")


def morse_of(text):
    out = []
    for ch in text.upper():
        m = cw.INV.get(ch)
        if m:
            out.append(m)
    return out


def elem_distance(a, b):
    """Edit distance between two tokens measured in MORSE ELEMENTS -
    the unit in which CW errors actually occur."""
    ma, mb = "|".join(morse_of(a)), "|".join(morse_of(b))
    if not ma or not mb:
        return 99
    la, lb = len(ma), len(mb)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ma[i - 1] != mb[j - 1]))
        prev = cur
    return prev[lb]


MIN_ELEMS = 6          # a token shorter than this cannot be repaired
MAX_COST_FRAC = 0.25   # repair must change < this fraction of the elements


def repair_token(tok, max_cost=2):
    """Nearest lexicon entry, or None. Three guards added 8/03 after the
    first run turned `EN F S DR ME` into `R R R DN GE` - short tokens sit
    within one element of half the lexicon, so an unguarded search
    launders noise into plausible words:

      * length floor: <6 morse elements is unrepairable (E, T, A, N are
        1-3 elements - everything is 'close' to them),
      * relative cost: a repair may change at most 25% of the elements,
      * ambiguity veto: if two lexicon entries tie at the best cost, the
        evidence does not pick one, so we refuse rather than guess.
    """
    tok = tok.upper()
    if tok in PROSIGNS or CALLSIGN.match(tok):
        return tok, 0, "already-valid"
    n_elem = len("|".join(morse_of(tok)))
    if n_elem < MIN_ELEMS:
        return None, None, None
    scored = []
    for w in PROSIGNS:
        if abs(len(w) - len(tok)) > 2:
            continue
        scored.append((elem_distance(tok, w), w))
    if not scored:
        return None, None, None
    scored.sort()
    bc, best = scored[0]
    if bc > max_cost or bc / max(n_elem, 1) > MAX_COST_FRAC:
        return None, None, None
    if len(scored) > 1 and scored[1][0] == bc:      # ambiguous
        return None, None, None
    return best, bc, "lexicon"


def consensus(tokens, max_cost=3):
    """Merge near-duplicate tokens (the operator repeating themselves).
    Returns [(merged, n_copies, spread)] - INTERNAL evidence."""
    groups = []
    for t in tokens:
        placed = False
        for g in groups:
            if elem_distance(t, g[0]) <= max_cost:
                g.append(t)
                placed = True
                break
        if not placed:
            groups.append([t])
    out = []
    for g in groups:
        if len(g) < 2:
            continue
        # the member closest to all others wins (medoid in element space)
        best = min(g, key=lambda x: sum(elem_distance(x, y) for y in g))
        out.append((best, len(g), max(elem_distance(best, y) for y in g)))
    return out


def repair_text(text, verbose=False):
    toks = [t for t in re.split(r"[^A-Z0-9?/]+", text.upper()) if t]
    repairs, kept = [], []
    for t in toks:
        r, cost, how = repair_token(t)
        if r and how == "lexicon" and cost > 0:
            repairs.append((t, r, cost))
            kept.append(r)
        elif r:
            kept.append(r)
        else:
            kept.append(t)
    cons = consensus([t for t in toks if len(t) >= 3])
    return {"raw": " ".join(toks), "repaired": " ".join(kept),
            "lexicon_repairs": repairs,
            "repeated": [(c[0], c[1]) for c in cons]}


def decode_capture(iq_file, hz, secs=None):
    p = HARVEST / iq_file
    n = None if secs is None else int(secs * FS)
    iq = cc._load(p, n_max=n)
    env, aud = cw.envelope2(iq, FS, float(hz), aud=8000, bw_hz=150)
    t = cw.decode_env_auto(env, aud)
    if isinstance(t, (list, tuple)):
        t = " ".join(str(x) for x in t)
    return (t or "").split("{")[0].strip()


def cmd_demo(args):
    n = 0
    for j in sorted(glob.glob(str(HARVEST / "cw_*.json"))):
        d = json.loads(Path(j).read_text())
        c = d.get("center") or {}
        if not c.get("is_cw"):
            continue
        t = c.get("text")
        t = " ".join(map(str, t)) if isinstance(t, list) else (t or "")
        t = t.split("{")[0].strip()
        if not t:
            continue
        r = repair_text(t)
        if not r["lexicon_repairs"] and not r["repeated"]:
            continue
        n += 1
        print(f"\n{Path(j).stem[:26]}  ({c.get('wpm', 0):.0f} wpm)")
        print(f"  raw      : {r['raw'][:76]}")
        print(f"  repaired : {r['repaired'][:76]}")
        if r["lexicon_repairs"]:
            print("  changes  : " + ", ".join(
                f"{a}->{b} (cost {c_})" for a, b, c_ in r["lexicon_repairs"][:6]))
        if r["repeated"]:
            print("  repeated : " + ", ".join(
                f"{w} x{k}" for w, k in r["repeated"][:4]))
        if n >= args.n:
            break
    print(f"\n{n} captures shown. Every change is labelled with its cost in "
          f"MORSE ELEMENTS; repeats are internal signal evidence.")
    return 0


def cmd_full(args):
    """Decode the WHOLE capture (we were only reading the first 6 s of
    files up to 10 minutes long - 12% of the corpus audio) and repair."""
    rows = []
    for j in sorted(glob.glob(str(HARVEST / "cw_*.json"))):
        d = json.loads(Path(j).read_text())
        c = d.get("center") or {}
        if not c.get("is_cw") or c.get("hz") is None:
            continue
        p = HARVEST / d.get("iq_file", "")
        if not p.is_file():
            continue
        secs = p.stat().st_size / 4 / FS
        if secs < args.min_secs:
            continue
        try:
            short = decode_capture(d["iq_file"], c["hz"], secs=6)
            full = decode_capture(d["iq_file"], c["hz"],
                                  secs=min(secs, args.cap_secs))
        except Exception as e:
            continue
        rs, rf = repair_text(short), repair_text(full)
        rows.append({"id": Path(j).stem, "secs": round(secs, 1),
                     "short_tokens": len(rs["raw"].split()),
                     "full_tokens": len(rf["raw"].split()),
                     "full_repaired": rf["repaired"][:300],
                     "repeated": rf["repeated"][:6]})
        print(f"{Path(j).stem[:24]:<26} {secs:>6.0f}s  "
              f"6s->{len(rs['raw'].split()):>3} tok   "
              f"full->{len(rf['raw'].split()):>4} tok")
        if rf["repeated"]:
            print(f"    repeated: " + ", ".join(f"{w} x{k}"
                                                for w, k in rf["repeated"][:5]))
        if len(rows) >= args.n:
            break
    out = HERE.parent / "lab" / "full_decodes.json"
    out.write_text(json.dumps(rows, indent=1))
    if rows:
        st = sum(r["short_tokens"] for r in rows)
        ft = sum(r["full_tokens"] for r in rows)
        print(f"\n{len(rows)} captures: {st} tokens from 6 s -> {ft} tokens "
              f"from full length ({ft/max(st,1):.1f}x more text)")
        print(f"written to {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--n", type=int, default=12)
    f = sub.add_parser("full")
    f.add_argument("--n", type=int, default=25)
    f.add_argument("--min-secs", type=float, default=30.0)
    f.add_argument("--cap-secs", type=float, default=120.0)
    a = ap.parse_args()
    sys.exit(cmd_demo(a) if a.cmd == "demo" else cmd_full(a))


if __name__ == "__main__":
    main()
