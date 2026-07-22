#!/usr/bin/env python3
"""Deprecated entry — delegates to stack_cube_expert_policy (backward compatibility)."""

from integrations.isaac_lab.scripts.stack_cube_expert_policy import main

if __name__ == "__main__":
    raise SystemExit(main())
