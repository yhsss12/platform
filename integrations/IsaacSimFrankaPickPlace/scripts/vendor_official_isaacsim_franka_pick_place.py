"""
Optional helper: copy the official standalone example path information into a manifest.
This script does not copy NVIDIA source files by default.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--isaacsim-root", required=True)
parser.add_argument("--out-dir", default="vendor_manifest")
args = parser.parse_args()
root = Path(args.isaacsim_root)
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
example = root / "standalone_examples/api/isaacsim.robot.experimental.manipulators/franka/pick_place.py"
manifest = {
    "official_example_exists": example.exists(),
    "official_example_path": str(example),
    "official_controller_import": "isaacsim.robot.experimental.manipulators.examples.franka.FrankaPickPlace",
    "note": "Do not vendor NVIDIA source unless your license policy allows it; use runtime import instead."
}
(out / "OFFICIAL_ISAACSIM_FRANKA_PICK_PLACE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
