#!/usr/bin/env python3
"""P9-A: Build NutAssembly-PINN v1 repair training dataset from MimicGen demos."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
_INTEGRATION = _REPO / "integrations" / "NutAssemblyMimicGen"
_DATA_ROOT = Path(os.environ.get("EAI_DATA_ROOT") or (_REPO / "eai-data")).expanduser()
if str(_INTEGRATION) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION))

from utils.pinn_repair_v1 import (  # noqa: E402
    XY_OFFSETS_M,
    apply_eef_perturbation,
    build_delta_vector,
    build_feature_vector,
    extract_align_insert_segment,
)

DEFAULT_OUTPUT = _DATA_ROOT / "runs/nut_assembly/pinn_training"
DEFAULT_OFFICIAL = _DATA_ROOT / "assets/datasets/mimicgen/nut_assembly/source/nut_assembly.hdf5"
DEFAULT_P8_RAW = (
    _DATA_ROOT
    / "runs/nut_assembly/jobs/na_gen_p8_pinn_20260703_181533_97c8/datasets/nut_assembly_mimicgen_raw.hdf5"
)
DEFAULT_P8_SYNTH = (
    _DATA_ROOT
    / "runs/nut_assembly/jobs/na_gen_p8_pinn_20260703_181533_97c8/repair/candidates/synthetic_demo_1_1.hdf5"
)


def _demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(k for k in data_group.keys() if k.startswith("demo_"))


def _collect_sources(args: argparse.Namespace) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    if args.official_source.is_file():
        out.append(("official_mimicgen_source", args.official_source))
    if args.mimicgen_raw.is_file():
        out.append(("mimicgen_datagen_raw", args.mimicgen_raw))
    if args.synthetic_candidate.is_file():
        out.append(("synthetic_perturbation", args.synthetic_candidate))
    for extra in args.extra_hdf5:
        path = Path(extra)
        if path.is_file():
            out.append((f"extra_{path.stem}", path))
    return out


def _build_samples_from_hdf5(
    *,
    source_label: str,
    hdf5_path: Path,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    features: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    previews: list[dict[str, Any]] = []

    with h5py.File(hdf5_path, "r") as f:
        data = f.get("data")
        if data is None:
            return features, deltas, previews
        for demo_key in _demo_keys(data):
            demo_grp = data[demo_key]
            try:
                segment = extract_align_insert_segment(demo_grp)
            except KeyError:
                continue
            clean_eef = segment["eef_seg"]
            for xy_offset_m in XY_OFFSETS_M:
                angle = rng.uniform(0, 2 * np.pi)
                xy_off = (xy_offset_m * np.cos(angle), xy_offset_m * np.sin(angle))
                z_off = float(rng.uniform(-0.005, 0.005))
                perturbed = apply_eef_perturbation(
                    clean_eef,
                    xy_offset=xy_off,
                    z_offset=z_off,
                    action_noise=0.002,
                    rng=rng,
                )
                feat = build_feature_vector(segment, perturbed_eef=perturbed, xy_offset_m=xy_offset_m)
                delta = build_delta_vector(clean_eef, perturbed)
                features.append(feat)
                deltas.append(delta)
                previews.append(
                    {
                        "source": source_label,
                        "sourcePath": str(hdf5_path),
                        "demoKey": demo_key,
                        "xy_offset_m": xy_offset_m,
                        "xy_offset": [float(xy_off[0]), float(xy_off[1])],
                        "z_offset_m": z_off,
                        "featureDim": int(feat.shape[0]),
                        "deltaNorm": float(np.linalg.norm(delta)),
                        "repairStages": ["align_over_peg", "descend_insert"],
                    }
                )
    return features, deltas, previews


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NutAssembly PINN repair training dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--official-source", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--mimicgen-raw", type=Path, default=DEFAULT_P8_RAW)
    parser.add_argument("--synthetic-candidate", type=Path, default=DEFAULT_P8_SYNTH)
    parser.add_argument("--extra-hdf5", type=Path, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sources = _collect_sources(args)
    if not sources:
        raise SystemExit("No training HDF5 sources found")

    all_features: list[np.ndarray] = []
    all_deltas: list[np.ndarray] = []
    all_previews: list[dict[str, Any]] = []
    source_records: list[str] = []

    for label, path in sources:
        feats, dlt, previews = _build_samples_from_hdf5(source_label=label, hdf5_path=path, rng=rng)
        all_features.extend(feats)
        all_deltas.extend(dlt)
        all_previews.extend(previews)
        source_records.append(f"{label}:{path}")

    if not all_features:
        raise SystemExit("No training samples generated")

    features_arr = np.stack(all_features, axis=0).astype(np.float32)
    deltas_arr = np.stack(all_deltas, axis=0).astype(np.float32)
    npz_path = args.output_dir / "repair_training_dataset.npz"
    np.savez_compressed(
        npz_path,
        features=features_arr,
        trajectory_delta=deltas_arr,
        corrected_eef=deltas_arr,
        target_success_delta=deltas_arr,
    )

    manifest = {
        "sampleCount": int(features_arr.shape[0]),
        "sources": source_records,
        "segmentLen": 48,
        "actionDim": 7,
        "featureDim": int(features_arr.shape[1]),
        "deltaDim": int(deltas_arr.shape[1]),
        "xyOffsetsM": list(XY_OFFSETS_M),
        "repairStages": ["align_over_peg", "descend_insert"],
        "inputSchema": "nut_assembly_repair_v1",
        "outputSchema": "trajectory_delta_v1",
    }
    manifest_path = args.output_dir / "repair_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    preview_path = args.output_dir / "samples_preview.json"
    preview_path.write_text(json.dumps(all_previews[:20], indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[p9-a] wrote {npz_path} samples={features_arr.shape[0]}")
    print(f"[p9-a] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
