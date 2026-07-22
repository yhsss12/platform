#!/usr/bin/env bash
set -e

# Run from the IsaacLab repository root.
# This uses SkillGen during dataset generation.

./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
  --device cpu \
  --num_envs 10 \
  --generation_num_trials 100 \
  --use_skillgen \
  --input_file ./datasets/stack_cube_seed_annotated.hdf5 \
  --output_file ./datasets/stack_cube_skillgen_generated.hdf5

