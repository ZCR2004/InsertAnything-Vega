"""Pinned source paths; do not import the Isaac Lab extension for inference."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STEADYHAND = ROOT / "third_party/steadyhand"
SIM2REAL = ROOT / "source/InsertAnything/InsertAnything/tasks/direct/insertanything/sim2real"


def bootstrap():
    for path in (STEADYHAND, SIM2REAL):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


bootstrap()
