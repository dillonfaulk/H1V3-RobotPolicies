"""Push mechanism for testing the fall trigger -- reuses the stock
`isaaclab.envs.mdp.events.push_by_setting_velocity` directly (the exact mechanism already used
throughout this project's own training), not a reimplementation.

OpenHorizon's own validated push magnitude: initial velocity 0.8-2.0 m/s from standing.
"""

from __future__ import annotations

import torch
from isaaclab.envs.mdp.events import push_by_setting_velocity

DEFAULT_VELOCITY_RANGE = {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}


def apply_push(env, velocity_range: dict = None) -> None:
    velocity_range = velocity_range or DEFAULT_VELOCITY_RANGE
    env_ids = torch.tensor([0], device=env.device)
    push_by_setting_velocity(env, env_ids, velocity_range)
