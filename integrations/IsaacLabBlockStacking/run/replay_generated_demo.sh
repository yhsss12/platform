#!/usr/bin/env bash
set -e

# Run from the IsaacLab repository root.
# This replays generated HDF5 demonstrations.

./isaaclab.sh -p scripts/tools/replay_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --dataset_file ./datasets/stack_cube_generated.hdf5

