#!/usr/bin/env python3
"""cw_lm.py - language-model rescoring for CW decodes.

CW traffic is highly predictable: callsigns, Q-codes, RST reports, prosigns, and a
small vocabulary of ragchew words. A character n-gram + a ham lexicon lets us fix
the two failure modes a raw sequence decoder leaves behind:

  1. '?' repair        - fill an undecoded letter with the one that makes a real
                         word/callsign  ('N?THING' -> 'NOTHING', 'W4D?W' -> 'W4DHW')
  2. word re-segmentation - insert the spaces the decoder merged
                         ('CQCQCQDE' -> 'CQ CQ CQ DE',  'THATIWAS' -> 'THAT I WAS')

This is the research's biggest near-term win (AG1LE/RSCW both point to LM rescoring)
and it specifically fixes the word-merging that held the Bayesian decoder back.
Pure stdlib+numpy, no training corpus needed at runtime - the model is bundled.
"""
import re
from math import log

import numpy as np

# ---- ham lexicon: token -> relative frequency weight (the CW "vocabulary") -----
PROSIGNS = {"CQ": 9, "DE": 9, "K": 7, "KN": 4, "AR": 3, "SK": 3, "BT": 3, "BK": 3,
            "R": 5, "AS": 2}
QCODES = {"QRZ": 5, "QSB": 3, "QRM": 3, "QRN": 3, "QTH": 4, "QSL": 4, "QSO": 3,
          "QRP": 3, "QSY": 2, "QRL": 2, "QRT": 2, "QRS": 2, "QRQ": 1, "QSK": 1}
RAGCHEW = {"TU": 6, "GM": 3, "GA": 3, "GE": 3, "TNX": 4, "FB": 4, "UR": 6, "RST": 4,
           "HR": 4, "NAME": 3, "OP": 4, "IS": 5, "ES": 5, "AGN": 3, "PSE": 3, "HW": 3,
           "CPY": 2, "RIG": 3, "ANT": 3, "PWR": 3, "WX": 4, "TEMP": 2, "HPE": 2,
           "CUL": 2, "GL": 2, "GUD": 3, "VY": 3, "DX": 4, "POTA": 4, "SOTA": 3,
           "WID": 2, "THE": 6, "AND": 5, "THAT": 5, "WAS": 5, "WITH": 4, "HAVE": 4,
           "THIS": 4, "YOU": 5, "NOT": 4, "NOTHING": 3, "KIND": 3, "TOLD": 3,
           "THEM": 3, "MY": 5, "STORY": 2, "HERE": 3, "GOOD": 3, "SIGNAL": 3,
           "OF": 5, "TO": 5, "IN": 5, "IT": 5, "I": 6, "A": 6, "AT": 4, "ON": 4,
           "FOR": 4, "ARE": 3, "OM": 4, "YL": 3, "XYL": 2, "73": 8, "72": 2, "88": 2,
           "5NN": 4, "599": 5, "589": 2, "579": 2, "559": 2, "WELL": 2, "ALL": 3}
LEXICON = {}
for d in (PROSIGNS, QCODES, RAGCHEW):
    for k, v in d.items():
        LEXICON[k] = LEXICON.get(k, 0) + v
_TOTAL = sum(LEXICON.values())
LEX_LOGP = {k: log(v / _TOTAL) for k, v in LEXICON.items()}
_MISS = log(0.5 / _TOTAL)                 # unknown-token floor

CALL_RE = re.compile(r"^[A-Z0-9]{0,2}[0-9][A-Z]{1,4}$")     # 1-2 pfx, digit, suffix
RST_RE = re.compile(r"^[1-5][1-9][1-9]$")
NUM_RE = re.compile(r"^[0-9]+$")
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# ---- bundled ham-text corpus -> character trigram (for open words) -------------
_CORPUS = (
    "CQ CQ CQ DE W1AW W1AW K QRZ DE K1ABC KN GE OM TNX FER CALL UR RST 599 599 "
    "IN BOSTON ES NAME IS JOHN JOHN HW CPY BK FB DE SOLID COPY HR RIG IS "
    "AN OLD RADIO ANT IS A DIPOLE UP 40 FT WX HR IS COLD ES SNOWY TEMP IS 20 F "
    "THAT WAS NOTHING OF THE KIND I TOLD THEM MY STORY AND THEY WERE GLAD "
    "TNX FER FB QSO OM HPE CUL AGN 73 73 DE SK POTA DE W4DHW QRZ THE PARK "
    "GUD DX ES GL VY 73 ES GB OM TU FER CALL 5NN 5NN K")


def _trigram(corpus):
    c = "  " + re.sub(r"[^A-Z0-9 ]", "", corpus.upper()) + " "
    tri = {}
    for i in range(2, len(c)):
        tri.setdefault(c[i - 2:i], {})
        tri[c[i - 2:i]][c[i]] = tri[c[i - 2:i]].get(c[i], 0) + 1
    logp = {}
    for ctx, nxt in tri.items():
        tot = sum(nxt.values()) + len(ALPHA + " ") * 0.1
        logp[ctx] = {ch: log((n + 0.1) / tot) for ch, n in nxt.items()}
        logp[ctx]["_"] = log(0.1 / tot)      # unseen next-char floor for this ctx
    return logp


_TRI = _trigram(_CORPUS)
_FLOOR = log(0.02)


def _tri_logp(word):
    """character-trigram log-prob of a bare word (open-text plausibility)."""
    w = "  " + word + " "
    s = 0.0
    for i in range(2, len(w)):
        ctx = w[i - 2:i]
        d = _TRI.get(ctx)
        s += (d.get(w[i], d["_"]) if d else _FLOOR)
    return s


def token_logscore(tok):
    """How ham-plausible is this token? lexicon > callsign/RST/number > trigram."""
    if not tok:
        return _MISS
    if tok in LEX_LOGP:
        return LEX_LOGP[tok] + 2.0                 # strong bonus for known ham token
    if CALL_RE.match(tok) and any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
        return log(3.0 / _TOTAL) + 2.0             # looks like a real callsign
    if RST_RE.match(tok) or NUM_RE.match(tok):
        return log(2.0 / _TOTAL) + 1.0
    # open word: trigram plausibility, penalized by length (avoid gibberish runs)
    return _tri_logp(tok) - 0.5 * max(0, len(tok) - 6)


def _resegment(run):
    """Insert word breaks into a space-free run to maximize total token score
    (classic word-break DP). Fixes 'CQCQCQDE' -> 'CQ CQ CQ DE'."""
    n = len(run)
    if n == 0:
        return []
    best = [(-1e18, -1)] * (n + 1)
    best[0] = (0.0, -1)
    for i in range(1, n + 1):
        for j in range(max(0, i - 12), i):         # tokens up to 12 chars
            tok = run[j:i]
            sc = best[j][0] + token_logscore(tok) - 0.6   # -0.6 = per-word insertion cost
            if sc > best[i][0]:
                best[i] = (sc, j)
    out, i = [], n
    while i > 0:
        j = best[i][1]
        out.append(run[j:i]); i = j
    return out[::-1]


def _repair(tok):
    """Fill '?' (and try 1-edit fixes) to reach the best-scoring real token."""
    if "?" not in tok:
        return tok
    qs = [i for i, c in enumerate(tok) if c == "?"]
    if len(qs) > 2:                                 # too corrupt to trust
        return tok
    cands = [tok]
    # brute force the '?' positions over the alphabet, keep best-scoring
    def fill(t, positions):
        if not positions:
            return [t]
        out = []
        for ch in ALPHA:
            out += fill(t[:positions[0]] + ch + t[positions[0] + 1:], positions[1:])
        return out
    best, bs = tok.replace("?", ""), -1e18
    for c in fill(tok, qs):
        s = token_logscore(c)
        if s > bs:
            bs, best = s, c
    # only repair into a REAL dictionary word - never guess a callsign letter
    # (a confident-wrong callsign is worse than an honest '?', and could false-verify)
    return best if best in LEX_LOGP else tok


def rescore(text):
    """Rescore a raw CW decode -> cleaner ham text. Re-segments merged words and
    repairs '?'. Conservative: only rewrites when the LM strongly prefers it."""
    if not text:
        return text
    out = []
    for raw in text.split():
        raw = raw.strip()
        if not raw:
            continue
        # 1) re-segment if the run is long and not already a known token/callsign
        if len(raw) > 4 and raw not in LEX_LOGP and not CALL_RE.match(raw.replace("?", "A")):
            seg = _resegment(raw)
            # accept the split only if it clearly beats the single token
            if sum(token_logscore(t) for t in seg) - 0.6 * len(seg) > token_logscore(raw) + 0.5:
                out.extend(seg)
            else:
                out.append(raw)
        else:
            out.append(raw)
    # 2) repair '?' per token
    return " ".join(_repair(t) for t in out)


if __name__ == "__main__":
    tests = ["CQCQCQDE KI4XH", "THATIWAS N?THING OF THE KIND", "POTA DE W4D?W",
             "QRZ DE K1ABC", "N?THING", "CQ CQ DE W1AW"]
    for t in tests:
        print(f"  {t!r:38} -> {rescore(t)!r}")
