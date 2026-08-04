#!/usr/bin/env python3
"""cw_map.py - whole-band CW finder: every Morse transmission in one look.

Born 2026-08-03 from live user feedback: the per-carrier scanner (/scanbands)
said "cw=0" on a band where the user found keyed CW by EYE on the waterfall in
seconds (7026.6 kHz, 34 WPM, banked as lab/user_find_7026.wav). The radio
already delivers the whole 250 kHz span - hopping a probe across carriers one
at a time throws that away. This module instead fingerprints EVERY bin of a
spectrogram at once for the thing eyes key on: ON-OFF KEYING RHYTHM.

The fingerprint (per frequency bin, over a ~20 s window):
  * bimodal power  - a keyed carrier is clearly ON or clearly OFF, nothing
                     between (contrast = p85/p15 against the bin's own floor)
  * duty 15-85%    - Morse spends real time in both states; a steady carrier
                     or FT8 tone (~84% on over its 15 s cycle) fails this
  * transition rate 1.5-25 Hz - dits/dahs blink; FT8 flips ~0.1/s, RTTY is
                     continuous, lightning crashes are one-shot
  * run lengths    - median ON-run 25-400 ms = dit@45wpm .. dah@8wpm

Modes:
  python cw_map.py selftest              # synthetic band: CW vs FT8 vs carrier
  python cw_map.py file cap.cs16 [fs]    # map a capture (corpus validation)

Library: cw_map(iq, fs) -> list of {khz_off, score, wpm_est, duty, trans_hz}
sorted by score. Pure numpy, no GPU needed (the neural waterfall-detector is
task #52's later stage; this is the classic-DSP first pass).
"""
import sys
from pathlib import Path

import numpy as np

NFFT = 2048            # 122 Hz bins @ 250 kS/s - one CW signal ~= 1-2 bins
HOP = 2048             # 8.2 ms frames -> 122 fps, plenty for 45 wpm (27 ms dit)
ON_DB = 6.0            # a frame is ON if the bin is this far over its own floor


def _spectrogram(iq, fs):
    n = (len(iq) - NFFT) // HOP + 1
    if n < 100:
        return None, None
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    win = np.hanning(NFFT).astype(np.float32)
    sp = np.fft.fftshift(np.abs(np.fft.fft(iq[idx] * win, axis=1)), axes=1)
    freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, 1.0 / fs))
    return sp.astype(np.float32), freqs


def _runs(on):
    """Boolean frame series -> (on_run_lengths, transitions)."""
    d = np.diff(on.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if on[0]:
        starts = np.r_[0, starts]
    if on[-1]:
        ends = np.r_[ends, len(on)]
    return ends - starts, len(starts) + len(ends)


def cw_map(iq, fs, top=20):
    """Fingerprint every bin for Morse keying. Returns candidates sorted by
    score (0..1); khz_off is relative to the capture centre."""
    sp, freqs = _spectrogram(np.asarray(iq, np.complex64), fs)
    if sp is None:
        return []
    fps = fs / HOP
    secs = sp.shape[0] / fps
    out = []
    # noise floor = each bin's QUIET tail (p10), NOT its median: a dah-heavy
    # fist keys ON >50% of the time, which drags the median up to the signal
    # itself and blinds a 6-dB-over-median gate (caught by selftest, 8/03).
    floor = np.percentile(sp, 10, axis=0) + 1e-12
    snr = sp / floor[None, :]
    hot = np.flatnonzero(np.percentile(snr, 85, axis=0) > 10 ** (ON_DB / 20))
    for b in hot:
        s = snr[:, b]
        p85, p15 = np.percentile(s, 85), np.percentile(s, 15)
        contrast = p85 / max(p15, 1e-6)
        if contrast < 3.0:                         # keying is bimodal or it isn't
            continue                               # (noise flicker ~2.4, rejected)
        # ON gate = geometric midpoint of the two modes, per bin. A fixed
        # dB-over-floor gate sits inside the noise's own flicker and turns
        # every gap frame into a false ON (caught by selftest, 8/03).
        on = s > np.sqrt(p85 * p15)
        duty = float(on.mean())
        if not 0.15 <= duty <= 0.85:               # steady carriers + FT8 out
            continue
        runs, ntrans = _runs(on)
        trans_hz = ntrans / (2.0 * secs)
        if not 1.5 <= trans_hz <= 25.0:            # blink rate = the Morse tell
            continue
        med_run_ms = float(np.median(runs)) / fps * 1000.0
        if not 25.0 <= med_run_ms <= 400.0:        # dit@45wpm .. dah@8wpm
            continue
        # FSK veto (8/04 live calibration): RTTY's per-bin blink rate sneaks
        # under the transition gate, but FSK has a tell OOK can't fake - the
        # mark and space bins blink in ANTI-correlation. If a neighbor within
        # ~1 kHz blinks complementary to this bin, it's FSK, not a fist.
        z = on.astype(np.float32) - on.mean()
        fsk = False
        for nb in range(max(0, b - 8), min(snr.shape[1], b + 9)):
            if abs(nb - b) < 2:
                continue
            zn = (snr[:, nb] > np.sqrt(p85 * p15)).astype(np.float32)
            zn = zn - zn.mean()
            den = np.sqrt((z * z).sum() * (zn * zn).sum())
            if den > 1e-6 and float((z * zn).sum() / den) < -0.4:
                fsk = True
                break
        if fsk:
            continue
        # score: how far inside the Morse box each measure sits
        sc = min(1.0, (contrast - 3.0) / 10.0 + 0.4) \
            * (1.0 - abs(duty - 0.45) / 0.45) \
            * min(1.0, ntrans / 30.0)
        out.append({"khz_off": round(freqs[b] / 1e3, 3),
                    "score": round(float(sc), 3),
                    "wpm_est": round(1200.0 / max(med_run_ms, 1e-3)),
                    "duty": round(duty, 2),
                    "trans_hz": round(trans_hz, 1),
                    "snr_db": round(20 * np.log10(float(np.percentile(s, 85))), 1)})
    # collapse adjacent-bin duplicates of one keyed carrier: keep the local best
    out.sort(key=lambda r: -r["score"])
    kept = []
    for r in out:
        if all(abs(r["khz_off"] - k["khz_off"]) > 0.3 for k in kept):
            kept.append(r)
    return kept[:top]


# ==========================================================================
def _synth_band(fs=250_000.0, secs=20.0, seed=7):
    """Synthetic band: 3 CW ops + FT8-like tone + steady carrier + noise."""
    rng = np.random.default_rng(seed)
    n = int(fs * secs)
    t = np.arange(n) / fs
    iq = (rng.normal(0, 1.0, n) + 1j * rng.normal(0, 1.0, n)).astype(np.complex64)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cw_synth
    cw_at = {}
    for off, wpm, txt in ((-60e3, 18, "CQ CQ DE W1ABC W1ABC K"),
                          (12.5e3, 28, "TEST DE K4RUM K4RUM"),
                          (95e3, 12, "VVV VVV DE N0XYZ")):
        env, efs = cw_synth.render(txt * 3, wpm=wpm, fs=2000.0, noise=0.0,
                                   jitter=0.02, fade=0.0, rise_ms=4.0, seed=1)
        key = np.interp(t, np.arange(len(env)) / efs, env).astype(np.float32)
        iq += (4.0 * key * np.exp(2j * np.pi * off * t)).astype(np.complex64)
        cw_at[round(off / 1e3, 1)] = wpm
    # FT8-like: 12.6 s steady tone, 2.4 s gap
    ft8 = ((t % 15.0) < 12.6).astype(np.float32)
    iq += (4.0 * ft8 * np.exp(2j * np.pi * 40e3 * t)).astype(np.complex64)
    iq += (4.0 * np.exp(2j * np.pi * -20e3 * t)).astype(np.complex64)  # carrier
    # RTTY-like FSK @ 45.45 bd, 340 Hz shift: mark/space blink complementary
    bit = (np.floor(t * 45.45) % 2).astype(np.float32)   # alternating idle
    iq += (4.0 * bit * np.exp(2j * np.pi * 70e3 * t)).astype(np.complex64)
    iq += (4.0 * (1 - bit) * np.exp(2j * np.pi * (70e3 + 340) * t)
           ).astype(np.complex64)
    return iq, cw_at


def cmd_selftest():
    print("cw_map selftest: 3 CW + FT8-like + steady carrier in 250 kHz noise")
    iq, cw_at = _synth_band()
    hits = cw_map(iq, 250_000.0)
    ok = True
    for off, wpm in sorted(cw_at.items()):
        got = [h for h in hits if abs(h["khz_off"] - off) < 0.3]
        tag = "OK" if got else "MISS"
        ok &= bool(got)
        est = got[0]["wpm_est"] if got else "-"
        print(f"  CW @ {off:+7.1f} kHz ({wpm} wpm): {tag}  est={est} wpm")
    false_pos = [h for h in hits
                 if all(abs(h["khz_off"] - o) >= 0.3 for o in cw_at)]
    for h in false_pos:
        print(f"  FALSE @ {h['khz_off']:+7.1f} kHz score={h['score']}")
    ok &= not false_pos
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_file(path, fs):
    raw = np.fromfile(path, np.int16).astype(np.float32) / 32768.0
    iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    print(f"{path}: {len(iq)/fs:.1f} s @ {fs/1e3:.0f} kS/s")
    for h in cw_map(iq, fs):
        print(f"  {h['khz_off']:+8.3f} kHz  score={h['score']:.2f} "
              f"~{h['wpm_est']:>2} wpm  duty={h['duty']}  "
              f"trans={h['trans_hz']}/s  snr={h['snr_db']} dB")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        sys.exit(cmd_selftest())
    if len(sys.argv) > 2 and sys.argv[1] == "file":
        fs = float(sys.argv[3]) if len(sys.argv) > 3 else 250_000.0
        sys.exit(cmd_file(sys.argv[2], fs))
    print(__doc__)
