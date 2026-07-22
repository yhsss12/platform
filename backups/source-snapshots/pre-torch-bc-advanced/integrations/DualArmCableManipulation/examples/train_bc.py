#!/usr/bin/env python3
"""Historical dual-arm torch-BC trainer snapshot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

OBS_KEYS = [
    "left_arm_joint_pos",
    "right_arm_joint_pos",
    "left_arm_joint_vel",
    "right_arm_joint_vel",
    "cable_state",
]


def _load_demo_group(demo_grp: h5py.Group, demo_key: str) -> tuple[np.ndarray, np.ndarray]:
    if "actions" not in demo_grp:
        raise KeyError(f"missing actions in {demo_key}")
    if "obs" not in demo_grp:
        raise KeyError(f"missing obs group in {demo_key}")

    obs_grp = demo_grp["obs"]
    parts: list[np.ndarray] = []
    for key in OBS_KEYS:
        if key not in obs_grp:
            raise KeyError(f"missing obs/{key} in {demo_key}")
        arr = np.asarray(obs_grp[key][:], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        parts.append(arr)

    obs = np.concatenate(parts, axis=-1)
    actions = np.asarray(demo_grp["actions"][:], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    if actions.shape[0] != obs.shape[0]:
        raise ValueError(
            f"action/obs length mismatch in {demo_key}: actions={actions.shape[0]} obs={obs.shape[0]}"
        )
    if actions.shape[0] <= 0:
        raise ValueError(f"empty trajectory in {demo_key}")
    return obs, actions


def _load_dataset(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    demo_keys: list[str] = []

    with h5py.File(dataset_path, "r") as h5:
        if "data" not in h5:
            raise SystemExit("HDF5 missing data/ group; cannot train")
        data_grp = h5["data"]
        demo_keys = sorted(k for k in data_grp.keys() if str(k).startswith("demo_"))
        if not demo_keys:
            raise SystemExit("HDF5 has no demo_* groups; cannot train")
        for demo_key in demo_keys:
            obs, actions = _load_demo_group(data_grp[demo_key], demo_key)
            all_obs.append(obs)
            all_actions.append(actions)

    obs_arr = np.concatenate(all_obs, axis=0)
    act_arr = np.concatenate(all_actions, axis=0)
    return obs_arr, act_arr, demo_keys


def main() -> int:
    parser = argparse.ArgumentParser(description="Dual-arm cable torch BC trainer")
    parser.add_argument("--dataset", "--hdf5", dest="dataset", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        print(f"torch not available: {exc}", file=sys.stderr)
        return 2

    epochs = int(args.num_epochs if args.num_epochs is not None else args.epochs)
    lr = float(args.learning_rate if args.learning_rate is not None else args.lr)
    dataset_path = Path(args.dataset).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.is_file() or dataset_path.stat().st_size == 0:
        print(f"dataset not found or empty: {dataset_path}", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_arr, act_arr, demo_keys = _load_dataset(dataset_path)
    obs_dim = int(obs_arr.shape[-1])
    action_dim = int(act_arr.shape[-1])
    num_samples = int(obs_arr.shape[0])

    device = torch.device(
        "cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    )

    model = nn.Sequential(
        nn.Linear(obs_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, action_dim),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    obs_tensor = torch.from_numpy(obs_arr)
    act_tensor = torch.from_numpy(act_arr)
    loader = DataLoader(
        TensorDataset(obs_tensor, act_tensor),
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
    )

    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)
            pred = model(batch_obs)
            loss = criterion(pred, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        final_loss = epoch_loss / max(batches, 1)
        print(f"Epoch {epoch} Loss: {final_loss:.6f}")

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "obs_keys": list(OBS_KEYS),
            "backend": "torch_bc",
        },
        ckpt_path,
    )

    metrics: dict[str, Any] = {
        "train_loss": final_loss,
        "num_samples": num_samples,
        "action_dim": action_dim,
        "obs_dim": obs_dim,
        "epochs": epochs,
        "demo_keys": demo_keys,
        "obs_keys": list(OBS_KEYS),
        "backend": "torch_bc",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    training_manifest = {
        "backendType": "torch_bc",
        "taskType": "dual_arm_cable_manipulation",
        "taskTemplateId": "dual_arm_cable_manipulation",
        "modelType": "bc",
        "actionDim": action_dim,
        "obsDim": obs_dim,
        "observationSchema": "dual_arm_cable_il_v1",
        "actionSchema": "dual_arm_bimanual_action_v1",
        "actionSemantics": "recorded_joint_position_targets",
        "checkpointPath": str(ckpt_path),
        "datasetPath": str(dataset_path),
        "metricsPath": str(out_dir / "metrics.json"),
    }
    (out_dir / "training_manifest.json").write_text(
        json.dumps(training_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("checkpoint:", ckpt_path)
    print("metrics:", out_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
