#!/usr/bin/env python3
"""docs_guard.py - keep the mermaid charts honest and un-breakable.

Two jobs:
 1. LINT every ```mermaid block in the given .md files for the traps that
    silently break rendering (GitHub shows an error box instead of a chart):
    unbalanced quotes/brackets, raw < > & inside quoted labels that aren't
    HTML entities or <br/>, the '-- >' arrow typo, unclosed fences.
    (Render-recipe law: html labels must HTML-escape specials.)
 2. DRIFT check (hamTuna): every endpoint the panel actually serves
    (u.path == "/x" in tools/panel.py) must be mentioned in docs/CONTROLS.md -
    an uncharted endpoint is how the telemetry contract rots.

  python tools/docs_guard.py                 # hamTuna defaults (lint + drift)
  python tools/docs_guard.py file1.md ...    # lint arbitrary chart files

Exit 1 on any error so it can gate a test run.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FENCE = re.compile(r"^```mermaid\s*$")
END = re.compile(r"^```\s*$")
ENTITY = re.compile(r"&[a-zA-Z]+[0-9]*;|&#\d+;")
ARROW_TYPO = re.compile(r"--\s+>")
ENDPOINT = re.compile(r'u\.path\s*==\s*"(/[a-z_0-9]+)"')


FLAG_NODE = re.compile(r"\w+>\s*\"")           # id>"label"] = mermaid flag shape
ALLOWED_TAGS = re.compile(r"</?(?:br|b|i|u|sub|sup)\s*/?>", re.I)


def lint_block(lines, start, path, errs, warns):
    for ln_no, raw in lines:
        ln = raw.strip()
        if ln.startswith("%%"):                # mermaid comment - ignored by renderer
            continue
        if ARROW_TYPO.search(ln):              # '-- >' renders as text, not an edge
            errs.append(f"{path}:{ln_no}: arrow typo '-- >'")
        if "�" in raw:                    # actual replacement char = encoding rot
            errs.append(f"{path}:{ln_no}: U+FFFD replacement char (encoding rot)")
        # the id>"..."] flag-node shape opens with '>' - credit its closing ']'
        depth = -len(FLAG_NODE.findall(ln))
        in_q = False
        qbuf = []
        for ch in ln:
            if ch == '"':
                if in_q:
                    label = "".join(qbuf)
                    stripped = ALLOWED_TAGS.sub("", label)
                    stripped = ENTITY.sub("", stripped)
                    for bad in "<>&":
                        if bad in stripped:
                            warns.append(f"{path}:{ln_no}: raw '{bad}' in label "
                                         f"\"{label[:40]}\" (escaping is safer)")
                            break
                    qbuf = []
                in_q = not in_q
            elif in_q:
                qbuf.append(ch)
            elif ch in "[({":
                depth += 1
            elif ch in "])}":
                depth -= 1
        if in_q:
            errs.append(f"{path}:{ln_no}: unbalanced quote")
        if depth > 0:                          # net-negative can be a shape form; net-open = broken
            errs.append(f"{path}:{ln_no}: unbalanced brackets ({depth:+d})")


def lint_file(path):
    errs, warns = [], []
    lines = path.read_text(encoding="utf-8").splitlines()
    in_block = False
    block = []
    n_blocks = 0
    for i, ln in enumerate(lines, 1):
        if not in_block and FENCE.match(ln):
            in_block, block = True, []
            n_blocks += 1
        elif in_block and END.match(ln):
            lint_block(block, i, path.name, errs, warns)
            in_block = False
        elif in_block:
            block.append((i, ln))
    if in_block:
        errs.append(f"{path.name}: unclosed ```mermaid fence")
    return n_blocks, errs, warns


def drift_check():
    """hamTuna panel endpoints vs CONTROLS.md mentions."""
    errs = []
    panel = HERE / "panel.py"
    controls = HERE.parent / "docs" / "CONTROLS.md"
    if not (panel.exists() and controls.exists()):
        return errs
    served = set(ENDPOINT.findall(panel.read_text(encoding="utf-8")))
    doc = controls.read_text(encoding="utf-8")
    for ep in sorted(served):
        if ep not in doc:
            errs.append(f"DRIFT: panel serves {ep} but docs/CONTROLS.md never "
                        f"mentions it - chart the control or it WILL get broken")
    return errs


def main():
    targets = ([Path(a) for a in sys.argv[1:]] if len(sys.argv) > 1
               else [HERE.parent / "docs" / "CONTROLS.md"])
    total_errs, total_warns = [], []
    for t in targets:
        if not t.exists():
            total_errs.append(f"{t}: missing")
            continue
        n, errs, warns = lint_file(t)
        print(f"[guard] {t.name}: {n} chart(s), {len(errs)} error(s), "
              f"{len(warns)} warning(s)")
        total_errs += errs
        total_warns += warns
    if len(sys.argv) == 1:                      # default run includes drift
        d = drift_check()
        print(f"[guard] endpoint drift: {len(d)} issue(s)")
        total_errs += d
    for e in total_errs:
        print("  !!", e)
    for w in total_warns[:8]:
        print("  ~ ", w)
    if len(total_warns) > 8:
        print(f"  ~  ... +{len(total_warns) - 8} more warnings")
    print(f"[guard] {'CLEAN' if not total_errs else 'FAIL'} "
          f"(errors gate, warnings advise)")
    sys.exit(1 if total_errs else 0)


if __name__ == "__main__":
    main()
