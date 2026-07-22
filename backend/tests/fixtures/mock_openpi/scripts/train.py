#!/usr/bin/env python3
"""Mock openpi train entry simulating official tyro CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

UNSUPPORTED_FLAGS = {
    "--config-path",
    "--learning-rate",
    "--output-dir",
}


def _reject_unsupported(unknown: list[str]) -> int:
    bad = [token for token in unknown if token in UNSUPPORTED_FLAGS or token.startswith("--config-path")]
    if bad:
        print(f"Unrecognized options: {' '.join(bad)}", file=sys.stderr)
        return 2
    if unknown:
        print(f"Unrecognized options: {' '.join(unknown)}", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", nargs="?", default="pi0_mock")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--checkpoint-base-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-train-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-wandb-enabled", action="store_true")
    parser.add_argument("--lr-schedule.peak-lr", dest="peak_lr", type=float, default=None)
    parser.add_argument("--lr-schedule.decay-lr", dest="decay_lr", type=float, default=None)

    args, unknown = parser.parse_known_args()
    reject_code = _reject_unsupported(unknown)
    if reject_code != 0:
        return reject_code

    total_steps = max(1, int(args.num_train_steps))
    learning_rate = args.peak_lr if args.peak_lr is not None else 2.5e-5
    ckpt_root = Path(args.checkpoint_base_dir).expanduser().resolve() / args.config_name / args.exp_name
    ckpt_root.mkdir(parents=True, exist_ok=True)

    for step in range(1, total_steps + 1):
        loss = 1.0 / step
        if step == 1 or step % max(1, args.log_interval) == 0 or step == total_steps:
            print(f"Step {step}: loss={loss:.4f}")

    step_dir = ckpt_root / str(total_steps)
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "params").write_bytes(b"MOCK_OPENPI_CHECKPOINT")
    manifest = {
        "format": "openpi_orbax_v1",
        "openpiBaseConfig": args.config_name,
        "expName": args.exp_name,
        "step": total_steps,
        "checkpointPath": str(step_dir),
        "learningRate": learning_rate,
        "seed": args.seed,
        "batchSize": args.batch_size,
    }
    (step_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"checkpoint: {step_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
