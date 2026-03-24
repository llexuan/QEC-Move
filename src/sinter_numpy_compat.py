"""Compatibility helpers for sinter on NumPy 2.

Notebook processes can import this module for documentation and parity with
previous runs. Worker processes used by sinter.collect do not execute notebook
imports, so the authoritative fix is patching sinter inside the venv.
"""

from __future__ import annotations

