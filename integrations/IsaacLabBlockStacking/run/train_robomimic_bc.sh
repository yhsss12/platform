#!/usr/bin/env bash
set -e

# Run from the IsaacLab repository root.
# This trains a BC policy from the generated HDF5 dataset.
# Config names may change across Isaac Lab versions; check official docs if this command needs adjustment.

./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --algo bc \
  --dataset ./datasets/stack_cube_generated.hdf5

