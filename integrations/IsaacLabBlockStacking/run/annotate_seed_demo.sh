#!/usr/bin/env bash
set -e

# Run from the IsaacLab repository root.
# This annotates seed demonstrations for Isaac Lab Mimic.

./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
  --device cpu \
  --auto \
  --input_file ./datasets/stack_cube_seed.hdf5 \
  --output_file ./datasets/stack_cube_seed_annotated.hdf5

