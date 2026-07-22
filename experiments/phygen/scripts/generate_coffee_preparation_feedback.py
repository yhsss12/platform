#!/usr/bin/env python3
"""Generate coffee_preparation PhyGen feedback jsonl from MimicGen source demos."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phygen.adapters.mimicgen.coffee_repair import main

if __name__ == "__main__":
    main()
