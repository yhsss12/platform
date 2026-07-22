import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.cable_threading.utils import DEFAULT_OBS_KEYS, make_env


def infer_obs_slices(obs_keys, metadata):
    env = make_env(
        robot=metadata.get("robot", "UR5e"),
        horizon=int(metadata.get("horizon", 250)),
        seed=int(metadata.get("seed", 0)),
        cable_model=metadata.get("cable_model", "rmb"),
        difficulty=metadata.get("difficulty", "easy"),
        has_renderer=False,
        has_offscreen_renderer=False,
    )
    try:
        obs = env.reset()
        dims = {key: int(np.asarray(obs[key]).size) for key in obs_keys}
    finally:
        env.close()

    start = 0
    slices = {}
    for key in obs_keys:
        end = start + dims[key]
        slices[key] = slice(start, end)
        start = end
    return slices


def split_episode_names(names, val_ratio=0.1, test_ratio=0.1):
    num_eps = len(names)
    num_test = int(round(num_eps * test_ratio))
    num_val = int(round(num_eps * val_ratio))
    if num_test + num_val >= num_eps and num_eps > 1:
        overflow = num_test + num_val - num_eps + 1
        num_test = max(0, num_test - overflow)
    num_train = max(1, num_eps - num_val - num_test) if num_eps else 0

    train = names[:num_train]
    val = names[num_train : num_train + num_val]
    test = names[num_train + num_val :]
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Input benchmark-native .npz dataset")
    parser.add_argument("--out", type=str, default="datasets/cable_threading/ur5e_rmb_scripted_train.hdf5")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=True)
    observations = dataset["observations"].astype(np.float32)
    next_observations = dataset["next_observations"].astype(np.float32)
    actions = dataset["actions"].astype(np.float32)
    rewards = dataset["rewards"].astype(np.float32)
    dones = dataset["dones"].astype(bool)
    episode_lengths = dataset["episode_lengths"].astype(np.int32)
    obs_keys = [str(key) for key in dataset["obs_keys"].tolist()]
    metadata = json.loads(str(dataset["metadata"]))
    episode_metadata = [json.loads(str(item)) for item in dataset["episode_metadata"]]

    if obs_keys != DEFAULT_OBS_KEYS:
        print("obs_keys:", obs_keys)

    obs_slices = infer_obs_slices(obs_keys, metadata)
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["env"] = metadata.get("env_name", "CableThreading")
        data_grp.attrs["env_args"] = json.dumps(metadata)
        data_grp.attrs["total"] = int(observations.shape[0])
        data_grp.attrs["success_semantics"] = metadata.get("success_semantics", "final_only")
        data_grp.attrs["obs_keys"] = json.dumps(obs_keys)

        start = 0
        demo_names = []
        for episode_idx, length in enumerate(episode_lengths.tolist()):
            end = start + length
            demo_name = f"demo_{episode_idx}"
            demo_names.append(demo_name)
            demo_grp = data_grp.create_group(demo_name)
            demo_grp.attrs["num_samples"] = int(length)
            demo_grp.attrs["benchmark_episode_metadata"] = json.dumps(episode_metadata[episode_idx])
            demo_grp.create_dataset("actions", data=actions[start:end])
            demo_grp.create_dataset("rewards", data=rewards[start:end])
            demo_grp.create_dataset("dones", data=dones[start:end].astype(np.int32))

            obs_grp = demo_grp.create_group("obs")
            next_obs_grp = demo_grp.create_group("next_obs")
            for key in obs_keys:
                obs_slice = obs_slices[key]
                obs_grp.create_dataset(key, data=observations[start:end, obs_slice])
                next_obs_grp.create_dataset(key, data=next_observations[start:end, obs_slice])
            start = end

        mask_grp = f.create_group("mask")
        train_names, val_names, test_names = split_episode_names(
            demo_names,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        mask_grp.create_dataset("train", data=np.asarray(train_names, dtype="S"))
        if val_names:
            mask_grp.create_dataset("valid", data=np.asarray(val_names, dtype="S"))
        if test_names:
            mask_grp.create_dataset("test", data=np.asarray(test_names, dtype="S"))

    print("saved_hdf5:", out_path)
    print("episodes:", len(episode_lengths))
    print("train_split:", len(train_names))
    print("valid_split:", len(val_names))
    print("test_split:", len(test_names))


if __name__ == "__main__":
    main()
