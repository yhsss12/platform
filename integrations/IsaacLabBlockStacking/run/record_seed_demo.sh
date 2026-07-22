#!/usr/bin/env bash
set -e

# Run from the IsaacLab repository root.
# This records seed demonstrations for the Stack Cube task.

./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --device cpu \
  --teleop_device keyboard \
  --dataset_file ./datasets/stack_cube_seed.hdf5 \
  --num_demos 10

