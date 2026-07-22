import argparse
import json
from pathlib import Path

import h5py
import numpy as np

DEFAULT_OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_gripper_qpos",
    "cable_end_pos",
    "pole_points",
    "endpoint_goal_pos",
    "attachment_state",
]


def dataset_has_mask(dataset_path, key):
    with h5py.File(Path(dataset_path).expanduser(), "r") as f:
        return f"mask/{key}" in f


def dataset_obs_keys(dataset_path):
    with h5py.File(Path(dataset_path).expanduser(), "r") as f:
        obs_keys_attr = f["data"].attrs.get("obs_keys", None)
        if obs_keys_attr is None:
            demos = f["data"]
            if not demos.keys():
                raise ValueError("dataset does not contain any demonstrations")
            first = demos[sorted(demos.keys())[0]].get("obs")
            if first is None:
                raise ValueError("dataset demonstration does not contain obs")
            available = sorted(str(key) for key in first.keys())
            if all(key in first for key in DEFAULT_OBS_KEYS):
                return list(DEFAULT_OBS_KEYS)
            return available
        if isinstance(obs_keys_attr, bytes):
            obs_keys_attr = obs_keys_attr.decode("utf-8")
        return [str(key) for key in json.loads(obs_keys_attr)]


def dataset_obs_modalities(dataset_path, obs_keys):
    """Infer robomimic modalities from the first demonstration's real tensors."""
    with h5py.File(Path(dataset_path).expanduser(), "r") as f:
        demos = f.get("data")
        if demos is None or not demos.keys():
            raise ValueError("dataset does not contain any demonstrations")
        demo = demos[sorted(demos.keys())[0]]
        obs = demo.get("obs")
        if obs is None:
            raise ValueError("dataset demonstration does not contain obs")

        rgb = []
        low_dim = []
        for key in obs_keys:
            if key not in obs:
                raise ValueError(f"observation key is missing from dataset: {key}")
            value = obs[key]
            sample_shape = tuple(value.shape[1:])
            is_rgb = (
                len(sample_shape) == 3
                and sample_shape[-1] in (3, 4)
                and (np.issubdtype(value.dtype, np.integer) or "image" in key.lower())
            )
            (rgb if is_rgb else low_dim).append(key)
    return {"rgb": rgb, "low_dim": low_dim}


def ensure_dataset_training_metadata(dataset_path):
    """Backfill standard Robomimic metadata for older generated datasets."""
    path = Path(dataset_path).expanduser().resolve()
    with h5py.File(path, "r+") as f:
        demos = f.get("data")
        if demos is None or not demos.keys():
            raise ValueError("dataset does not contain any demonstrations")

        first = demos[sorted(demos.keys())[0]].get("obs")
        if first is None:
            raise ValueError("dataset demonstration does not contain obs")
        if demos.attrs.get("obs_keys") is None:
            demos.attrs["obs_keys"] = json.dumps(sorted(str(key) for key in first.keys()))

        mask = f.require_group("mask")
        if "train" not in mask:
            train_demos = []
            has_validity_metadata = False
            for demo_name in sorted(demos.keys()):
                demo = demos[demo_name]
                valid = demo.attrs.get("valid_for_training")
                if valid is None:
                    valid = demo.attrs.get("success", demo.attrs.get("success_flag"))
                else:
                    has_validity_metadata = True
                if "success" in demo.attrs or "success_flag" in demo.attrs:
                    has_validity_metadata = True
                if bool(valid):
                    train_demos.append(demo_name)
            # Standard MimicGen output contains successful demonstrations only,
            # but older exports do not attach per-demo validity attributes.  In
            # that format the data group itself is the authoritative train set.
            if not train_demos and not has_validity_metadata:
                train_demos = sorted(demos.keys())
            if not train_demos:
                raise ValueError("dataset does not contain valid_for_training demonstrations")
            mask.create_dataset("train", data=np.asarray(train_demos, dtype="S"))
            print(f"backfilled train mask: {len(train_demos)} / {len(demos)} demos")


def prepare_resized_image_dataset(dataset_path, out_dir, rgb_keys, image_size):
    """Create a compact training view when collected camera frames are oversized."""
    source = Path(dataset_path).expanduser().resolve()
    size = int(image_size)
    with h5py.File(source, "r") as src:
        demos = src["data"]
        needs_resize = any(
            tuple(demo["obs"][key].shape[1:3]) != (size, size)
            for demo in demos.values()
            for key in rgb_keys
        )
    if not needs_resize:
        return source

    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"dataset_images_{size}.hdf5"
    if target.is_file() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return target

    from PIL import Image

    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(source, "r") as src, h5py.File(temporary, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value

        def copy_item(name, obj):
            if isinstance(obj, h5py.Group):
                group = dst.require_group(name)
                for key, value in obj.attrs.items():
                    group.attrs[key] = value
                return
            leaf = name.rsplit("/", 1)[-1]
            if "/obs/" not in f"/{name}" or leaf not in rgb_keys:
                src.copy(obj, dst.require_group(name.rsplit("/", 1)[0]), name=leaf)
                return
            frames = obj
            output = dst.create_dataset(
                name,
                shape=(frames.shape[0], size, size, frames.shape[-1]),
                dtype=frames.dtype,
                chunks=(1, size, size, frames.shape[-1]),
                compression="gzip",
                compression_opts=1,
            )
            for key, value in obj.attrs.items():
                output.attrs[key] = value
            for index in range(frames.shape[0]):
                output[index] = np.asarray(
                    Image.fromarray(frames[index]).resize((size, size), Image.Resampling.LANCZOS)
                )

        src.visititems(copy_item)
    temporary.replace(target)
    print(f"prepared resized image dataset: {source} -> {target} ({size}x{size})")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="robomimic-formatted hdf5 dataset")
    parser.add_argument("--out-dir", type=str, default="results/cable_threading_robomimic_bc")
    parser.add_argument("--name", type=str, default="robomimic_bc_lowdim")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-hidden-dims", type=str, default="512,512")
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--normalize-obs", action="store_true")
    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=0,
        help="checkpoint save interval; 0 means only save at final epoch when --save-final",
    )
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-data-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="optional platform model asset checkpoint")
    args = parser.parse_args()

    import torch
    from robomimic.config import config_factory
    from robomimic.scripts.train import train

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_dataset_training_metadata(args.dataset)
    obs_keys = dataset_obs_keys(args.dataset)
    modalities = dataset_obs_modalities(args.dataset, obs_keys)
    training_dataset = prepare_resized_image_dataset(
        args.dataset, out_dir, modalities["rgb"], args.image_size
    )

    config = config_factory("bc")
    config.unlock()
    config.experiment.name = args.name
    config.experiment.validate = dataset_has_mask(args.dataset, "valid")
    config.experiment.render = False
    config.experiment.render_video = False
    config.experiment.rollout.enabled = False
    config.experiment.save.enabled = bool(args.save_final or args.save_best or int(args.save_every_n_epochs) > 0)
    config.experiment.save.on_best_rollout_success_rate = False
    num_epochs = 2 if args.debug else int(args.num_epochs)
    if int(args.save_every_n_epochs) > 0:
        save_every_n_epochs = int(args.save_every_n_epochs)
    elif args.save_final:
        save_every_n_epochs = num_epochs
    else:
        save_every_n_epochs = max(1, num_epochs)
    config.experiment.save.every_n_epochs = save_every_n_epochs
    config.experiment.save.on_best_validation = bool(args.save_best) and config.experiment.validate

    config.train.data = str(training_dataset)
    config.train.output_dir = str(out_dir)
    config.train.batch_size = int(args.batch_size)
    config.train.num_epochs = num_epochs
    config.train.seed = int(args.seed)
    config.train.num_data_workers = int(args.num_data_workers)
    config.train.hdf5_cache_mode = "all"
    config.train.hdf5_use_swmr = True
    config.train.hdf5_load_next_obs = False
    config.train.hdf5_filter_key = "train"
    config.train.hdf5_validation_filter_key = "valid" if config.experiment.validate else None
    config.train.hdf5_normalize_obs = bool(args.normalize_obs)
    config.train.cuda = args.device != "cpu"

    config.algo.optim_params.policy.learning_rate.initial = float(args.learning_rate)
    config.algo.optim_params.policy.regularization.L2 = float(args.l2_regularization)
    actor_dims = [int(part.strip()) for part in str(args.actor_hidden_dims).split(",") if part.strip()]
    if len(actor_dims) < 2:
        actor_dims = [512, 512]
    config.algo.actor_layer_dims = [actor_dims[0], actor_dims[1]] if not args.debug else [256, 256]

    config.observation.modalities.obs.low_dim = modalities["low_dim"]
    config.observation.modalities.obs.rgb = modalities["rgb"]
    config.observation.modalities.obs.depth = []
    config.observation.modalities.obs.scan = []
    config.observation.modalities.goal.low_dim = []
    config.observation.modalities.goal.rgb = []
    config.observation.modalities.goal.depth = []
    config.observation.modalities.goal.scan = []

    has_valid_mask = config.experiment.validate
    if bool(args.normalize_obs) and has_valid_mask:
        print(
            "warning: hdf5_normalize_obs is incompatible with validation splits; "
            "disabling validation for this run."
        )
        config.experiment.validate = False
        config.train.hdf5_validation_filter_key = None
        config.experiment.save.on_best_validation = False

    config_path = out_dir / "config.json"
    config.dump(filename=str(config_path))

    ext_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
    config.lock()

    print("saved_config:", config_path)
    print("dataset:", config.train.data)
    print("output_dir:", config.train.output_dir)
    print("device:", args.device)
    print("obs_keys:", obs_keys)
    print("obs_modalities:", modalities)

    device_obj = torch.device(args.device)
    init_checkpoint = (args.init_checkpoint or "").strip() or None
    if init_checkpoint:
        from robomimic.utils.file_utils import load_dict_from_checkpoint
        import robomimic.algo as algo_mod

        ckpt = load_dict_from_checkpoint(init_checkpoint)
        original_factory = algo_mod.algo_factory

        def factory_with_init(*factory_args, **factory_kwargs):
            model = original_factory(*factory_args, **factory_kwargs)
            print("loading init checkpoint:", init_checkpoint)
            model.deserialize(ckpt["model"])
            return model

        algo_mod.algo_factory = factory_with_init
        try:
            train(config, device=device_obj)
        finally:
            algo_mod.algo_factory = original_factory
        return

    train(config, device=device_obj)


if __name__ == "__main__":
    main()
