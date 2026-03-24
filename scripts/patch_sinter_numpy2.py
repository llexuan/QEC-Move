"""Patch sinter for NumPy 2 integer scalar compatibility.

Usage:
    .venv/bin/python scripts/patch_sinter_numpy2.py
"""

from __future__ import annotations

from pathlib import Path
import site
import sys


TARGET_REL = Path("sinter/_decoding/_stim_then_decode_sampler.py")
OLD = "    return num_discards, num_errors"
NEW = "    return int(num_discards), int(num_errors)"


def _site_packages() -> list[Path]:
    out: list[Path] = []
    try:
        out.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass
    try:
        out.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    return out


def main() -> int:
    candidates = [p / TARGET_REL for p in _site_packages()]
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        print("Could not find sinter target file in site-packages.")
        return 1

    target = candidates[0]
    text = target.read_text(encoding="utf-8")
    if NEW in text:
        print(f"Already patched: {target}")
        return 0
    if OLD not in text:
        print(f"Expected text not found in {target}")
        return 2

    target.write_text(text.replace(OLD, NEW), encoding="utf-8")
    print(f"Patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

