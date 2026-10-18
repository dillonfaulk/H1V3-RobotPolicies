"""Observation assembly for both policies.

The recovery policy's 76-dim observation is exactly what the reused fall-recovery env's own
ObservationManager already computes (same task, same env, nothing to reassemble).

The walking policy's 81-dim observation has to be assembled BY HAND each step, since this script
reuses the fall-recovery task's env/scene (not the walking task's own gym env) as the single live
simulation host -- there's no command manager or AMP obs group here. Every underlying quantity is
pulled straight off the robot asset via the exact same stock `isaaclab.envs.mdp.observations`
functions the walking task's own ObservationsCfg uses (confirmed by direct source read, not
assumed): `base_lin_vel`/`base_ang_vel`/`projected_gravity` all use the asset's _b (body-frame,
not world-frame) tensors -- getting this wrong would silently feed the walking policy a
shape-correct but numerically wrong observation.

WALK_OBS_ORDER (81-dim, confirmed against asimov1_velocity_env_cfg.py's PolicyCfg):
    base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
    + joint_pos_rel(23) + joint_vel_rel(23) + last_action(23)

**WALKING_DEFAULT_JOINT_POS is NOT the reused env's own `default_joint_pos`.** Found via live
testing, not assumed: the walking task (`asimov1_velocity_amp_env_cfg.py`) overrides the robot's
init pose to a bent-knee "athletic" stance (`ASIMOV1_AMP_INIT_STATE_JOINT_POS`), while the reused
fall-recovery env's own default is all-zero (straight-leg standing). Since `joint_pos_rel` is
computed relative to whatever "default" is in scope, using the reused env's own (wrong) default
silently fed the walking policy an observation computed against the wrong reference pose -- a
systematic, large-magnitude error, not a minor one. Confirmed directly: even a textbook-perfect
clean standing start fell within 1.1s before this fix; see `rebase_walk_action` below for the
matching fix on the action side (the env's own ActionManager target formula has the exact same
problem, using the reused env's zero default as its offset instead of this one).
"""

from __future__ import annotations

import torch

# Confirmed by directly instantiating Isaac-Asimov1-Velocity-AMP-v0 and reading
# env.scene["robot"].data.default_joint_pos -- NOT the reused fall-recovery env's own default
# (all-zero). Same 23-joint order used everywhere else in this package (asset-intrinsic order,
# confirmed identical between both tasks' action terms).
WALKING_DEFAULT_JOINT_POS = [
    -0.13, 0.13, 0.0, 0.0, 0.0, 0.25, -0.25, 0.0, 0.0, 0.05, -0.05, 0.38, -0.38,
    0.0, 0.0, -0.26, 0.26, -0.4, 0.4, 0.0, 0.0, 0.0, 0.0,
]


def build_walk_obs(env, command_buf: torch.Tensor, last_action_walk: torch.Tensor) -> torch.Tensor:
    asset = env.scene["robot"]
    walking_default = torch.tensor([WALKING_DEFAULT_JOINT_POS], device=asset.data.joint_pos.torch.device)

    base_lin_vel = asset.data.root_lin_vel_b.torch
    base_ang_vel = asset.data.root_ang_vel_b.torch
    projected_gravity = asset.data.projected_gravity_b.torch
    joint_pos_rel = asset.data.joint_pos.torch - walking_default
    joint_vel_rel = asset.data.joint_vel.torch - asset.data.default_joint_vel.torch  # both tasks' default vel is 0

    return torch.cat(
        [base_lin_vel, base_ang_vel, projected_gravity, command_buf, joint_pos_rel, joint_vel_rel, last_action_walk],
        dim=-1,
    )


def get_recover_obs(env) -> torch.Tensor:
    """Thin wrapper -- the reused env IS the fall-recovery task's own env, so its ObservationManager
    already produces exactly the 76-dim vector the recovery policy expects, correctly ordered."""
    obs_dict = env.observation_manager.compute_group("policy")
    return obs_dict


def rebase_walk_action(
    env, raw_action: torch.Tensor, walk_action_scale: float, env_action_scale: float
) -> torch.Tensor:
    """Converts the walking policy's raw output into whatever needs to be PASSED to the reused
    env's own `env.step()` so the env's own ActionManager (which always applies
    `target = env's_own_default_joint_pos + passed_action * env_action_scale`, hardcoded to the
    fall-recovery task's values) resolves to the walking policy's TRUE intended absolute target,
    `target = WALKING_DEFAULT_JOINT_POS + raw_action * walk_action_scale`.

    Solves `env_default + passed * env_scale = WALKING_DEFAULT_JOINT_POS + raw * walk_scale` for
    `passed`. `env_action_scale` is passed in explicitly (the caller's own known constant) rather
    than introspected off the action term, to avoid depending on exactly how `JointPositionActionCfg`
    happens to store it internally.
    """
    asset = env.scene["robot"]
    env_default = asset.data.default_joint_pos.torch
    walking_default = torch.tensor([WALKING_DEFAULT_JOINT_POS], device=env_default.device)

    desired_target = walking_default + raw_action * walk_action_scale
    return (desired_target - env_default) / env_action_scale
