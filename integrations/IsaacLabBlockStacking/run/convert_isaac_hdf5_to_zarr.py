from pathlib import Path

import h5py
import numpy as np
import zarr


def _as_time_major_array(dataset):
    arr = np.asarray(dataset)
    if arr.ndim == 1:
        return arr.reshape(arr.shape[0], 1)
    return arr.reshape(arr.shape[0], -1)


def flatten_obs(obs_group):
    """Flatten low-dimensional Isaac observations into one state vector.

    The exact HDF5 keys can change with the Isaac Lab task and observation
    manager config. Keep the first version conservative: use whatever known
    low-dimensional keys exist, and fail loudly if none are present.
    """

    candidate_keys = [
        "joint_pos",
        "joint_vel",
        "eef_pos",
        "eef_quat",
        "gripper_pos",
        "object",
        "cube_positions",
        "cube_orientations",
        "proprio",
        "policy",
    ]

    obs_parts = []

    for key in candidate_keys:
        if key not in obs_group:
            continue

        value = obs_group[key]

        if isinstance(value, h5py.Group):
            nested_parts = []
            for nested_key in sorted(value.keys()):
                nested_parts.append(_as_time_major_array(value[nested_key]))
            if nested_parts:
                obs_parts.append(np.concatenate(nested_parts, axis=-1))
        else:
            obs_parts.append(_as_time_major_array(value))

    if not obs_parts:
        available = ", ".join(sorted(obs_group.keys()))
        raise RuntimeError(f"No supported obs keys found. Available keys: {available}")

    min_len = min(part.shape[0] for part in obs_parts)
    obs_parts = [part[:min_len] for part in obs_parts]
    return np.concatenate(obs_parts, axis=-1).astype(np.float32)


def convert_hdf5_to_zarr(hdf5_path, zarr_path):
    hdf5_path = Path(hdf5_path)
    zarr_path = Path(zarr_path)

    all_states = []
    all_actions = []
    episode_ends = []
    total_steps = 0

    with h5py.File(hdf5_path, "r") as hdf5_file:
        data_group = hdf5_file["data"]
        demo_names = sorted(data_group.keys())

        for demo_name in demo_names:
            demo = data_group[demo_name]
            states = flatten_obs(demo["obs"])
            actions = np.asarray(demo["actions"]).astype(np.float32)

            if actions.ndim == 1:
                actions = actions.reshape(actions.shape[0], 1)
            else:
                actions = actions.reshape(actions.shape[0], -1)

            length = min(states.shape[0], actions.shape[0])
            states = states[:length]
            actions = actions[:length]

            all_states.append(states)
            all_actions.append(actions)
            total_steps += length
            episode_ends.append(total_steps)

    states = np.concatenate(all_states, axis=0).astype(np.float32)
    actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    episode_ends = np.asarray(episode_ends, dtype=np.int64)

    root = zarr.open(str(zarr_path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")

    data.create_array(
        "state",
        data=states,
        chunks=(min(1024, states.shape[0]), states.shape[1]),
    )
    data.create_array(
        "action",
        data=actions,
        chunks=(min(1024, actions.shape[0]), actions.shape[1]),
    )
    meta.create_array("episode_ends", data=episode_ends)

    print("Converted Isaac HDF5 to Zarr.")
    print(f"states: {states.shape}")
    print(f"actions: {actions.shape}")
    print(f"episode_ends: {episode_ends.shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Isaac Lab Stack Cube HDF5 to Zarr.")
    parser.add_argument("--input", default="stack_cube_generated.hdf5", help="Input HDF5 path.")
    parser.add_argument("--output", default="isaac_stack_cube_dp.zarr", help="Output Zarr path.")
    cli_args = parser.parse_args()
    convert_hdf5_to_zarr(cli_args.input, cli_args.output)

