"""Expose the pinned SteadyHand runtime retained with this repository."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STEADYHAND = ROOT / "third_party/steadyhand"


def bootstrap():
    if str(STEADYHAND) not in sys.path:
        sys.path.insert(0, str(STEADYHAND))


bootstrap()
