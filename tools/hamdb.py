#!/usr/bin/env python3
"""hamdb.py — verify decoded callsigns against a real ham-license database.

Uses callook.info (US amateur licenses, free, no key). A VALID result means the
call belongs to a real licensed ham — so points are only awarded for calls we
can prove are real, not decode artifacts that happen to match the pattern.
Answers are cached to lab/callsign_cache.json so we hit the network once per call.

Non-US / DX calls come back UNKNOWN (callook is US-only) — logged as "heard,
unverified", never scored, and retried (a worldwide source can slot in later).
"""
import json
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "lab" / "callsign_cache.json"
_cache = {}


def _load():
    global _cache
    try:
        _cache = json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        _cache = {}


def _save():
    try:
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(_cache), encoding="utf-8")
    except Exception:
        pass


def verify(call):
    """-> {'status': VALID|INVALID|UNKNOWN, 'name': str, 'qth': str}."""
    call = call.upper().strip()
    if call in _cache:
        return _cache[call]
    try:
        r = json.loads(urllib.request.urlopen(
            f"https://callook.info/{call}/json", timeout=8).read())
        res = {"status": r.get("status", "UNKNOWN"),
               "name": (r.get("name") or "").title(),
               "qth": r.get("address", {}).get("line2", "")}
    except Exception:
        res = {"status": "UNKNOWN", "name": "", "qth": ""}
    if res["status"] in ("VALID", "INVALID"):   # cache only definitive answers
        _cache[call] = res
        _save()
    return res


_load()

if __name__ == "__main__":
    import sys
    for c in sys.argv[1:] or ["W1AW"]:
        print(c, "->", verify(c))
