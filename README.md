# hamTuna 📻🐟

**Adaptive amateur-radio decoding — the Tuna method on the ham bands.**

Part of the Tuna family:
[TV Tuna](https://github.com/Felbs/Software-TV-Tuner) →
[Radio Tuna](https://github.com/Felbs/gr-radiotuna) →
[wxTuna](https://github.com/Felbs/wxTuna) →
[aeroTuna](https://github.com/Felbs/aeroTuna) → hamTuna.
Same thesis every time: every decoder secretly knows how well it's
doing — surface that truth-dial and close the loop on it.

The ham bands are a mixed world — **digital** (APRS, CW, PSK31, FT8) and
**voice** (FM repeaters, SSB, and digital voice like DMR/D-STAR). One
repo, one tool per mode, shared plumbing.

## Quickstart
```bash
git clone https://github.com/Felbs/hamTuna.git
cd hamTuna
python tools/aprs.py selftest    # AX.25->AFSK->FM roundtrip, no radio needed
python tools/cw.py selftest      # Morse synth -> noise -> decode, no radio needed
```
**Dependencies:** `numpy`, `scipy`, and the `SoapySDR` python bindings +
a driver for your SDR (live modes only — selftests run with numpy/scipy
alone). Easiest path is [radioconda](https://github.com/ryanvolz/radioconda);
on Debian/Ubuntu: `apt install python3-numpy python3-scipy python3-soapysdr soapysdr-module-all`.

## Campaign 1 — APRS (`tools/aprs.py`)
Amateur radio's AIS: hams beacon callsign + position over AX.25 packets,
AFSK 1200 baud on 144.390 MHz (North America). Same HDLC framing and
CRC-16/X.25 truth dial we field-proved on maritime AIS; the Bell-202
modem underneath is new (non-coherent dual-tone detector).

```bash
python tools/aprs.py selftest              # full AX.25->AFSK->FM roundtrip, no radio
python tools/aprs.py capture --secs 120    # live: callsigns + positions heard
```

## Roadmap
- **CW (Morse)** — on-off keying with a confidence plane; the oldest
  digital mode meets the newest rescue tricks
- **Repeater activity scanner** — the voice-side inventory: which 2m/70cm
  repeaters are alive, learned by hour (the Knob of Time on voice)
- **PSK31** — narrowband keyboard chat
- **FT8** — via WSJT-X as the decode oracle; our value is the band-opening
  clock (see Radio Tuna's `hf_knob`), not re-decoding

## Hardware
Any SoapySDR-supported SDR + a VHF antenna. APRS beacons are strong and
bursty — patience beats gain.

## License
MIT
