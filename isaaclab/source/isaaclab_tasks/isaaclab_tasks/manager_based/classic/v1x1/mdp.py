# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""v1x1-specific MDP terms not covered by the reusable classic/humanoid mdp library.

Everything else (progress, alive, upright, move_to_target, etc.) is reused unmodified from
isaaclab_tasks.manager_based.classic.humanoid.mdp, same as Ant does. This module only adds terms that
need a v1x1-specific reason to exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp.rewards import GaitReward as _SpotGaitReward
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp.rewards import foot_clearance_reward  # noqa: F401

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
    from isaaclab.sensors import ContactSensor


def front_foot_forward_bonus(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, cap: float = 0.35
) -> torch.Tensor:
    """Reward the front feet for sitting forward of base_link, measured in the body's own local frame.

    Direct task-space reward (foot position relative to base_link) rather than an indirect joint-angle
    nudge -- doesn't care which combination of coxa/knee/ankle angles gets a foot out there, just rewards
    the outcome. This sidesteps the joint-sign-convention guesswork that made the equivalent joint-space
    fixes (front coxa/joint3 defaults) fragile and slow for the previous lineage to fully adopt. Clamped at
    `cap` so it can't reward pathological over-extension once a foot is already comfortably forward.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]  # (N, 2, 3)
    base_pos_w = asset.data.root_pos_w.torch[:, :3].unsqueeze(1)  # (N, 1, 3)
    base_quat_w = asset.data.root_quat_w.torch  # (N, 4)
    delta_w = foot_pos_w - base_pos_w  # (N, 2, 3)
    num_feet = delta_w.shape[1]
    quat_expanded = base_quat_w.unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4)
    delta_b = quat_apply_inverse(quat_expanded, delta_w.reshape(-1, 3)).reshape(delta_w.shape)
    forward_component = delta_b[..., 1]  # local +Y = this robot's anatomical forward
    return torch.sum(torch.clamp(forward_component, max=cap), dim=-1)


def left_right_symmetry_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize left/right joint asymmetry between mirrored leg pairs (leg1 vs leg2, leg3 vs leg4).

    Uses each joint's own empirically-established mirror convention (derived and verified live this
    session): joint1 (coxa) mirrors with an OPPOSITE sign between left/right legs (its effective direction
    depends on the mount-offset sign, which flips left/right); joints 2/3/4 mirror with the SAME sign (their
    rotation axes don't depend on that mount-offset sign the way joint1's does -- confirmed for joint2
    analytically and for joint3 by live correction after an initial wrong guess). Compares deviation from
    each joint's own default rather than raw angle, so it works regardless of the nonzero stance defaults.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch
    default = asset.data.default_joint_pos.torch
    dev = joint_pos - default
    names = asset.joint_names

    error = torch.zeros(env.num_envs, device=env.device)
    for joint, mirror_sign in (("joint1", -1.0), ("joint2", 1.0), ("joint3", 1.0), ("joint4", 1.0)):
        rear_l, rear_r = names.index(f"leg1_{joint}"), names.index(f"leg2_{joint}")
        front_l, front_r = names.index(f"leg3_{joint}"), names.index(f"leg4_{joint}")
        error = error + torch.square(dev[:, rear_l] - mirror_sign * dev[:, rear_r])
        error = error + torch.square(dev[:, front_l] - mirror_sign * dev[:, front_r])
    return error


def front_rear_symmetry_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize front/rear leg pairs (same side) using very different sized joint motions.

    Unlike left_right_symmetry_l2, this can't compare raw signed deviation directly -- the trot gait puts
    front and rear legs in different phases at any given instant (that's the whole point of trot_gait), so
    demanding matching signs at the same timestep would fight the gait pattern itself. Instead compares the
    MAGNITUDE of each joint's deviation from its own default, pushing front and rear legs toward a similarly
    sized range of motion without requiring them to move in lockstep.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch
    default = asset.data.default_joint_pos.torch
    dev = torch.abs(joint_pos - default)
    names = asset.joint_names

    error = torch.zeros(env.num_envs, device=env.device)
    for joint in ("joint1", "joint2", "joint3", "joint4"):
        front_l, rear_l = names.index(f"leg3_{joint}"), names.index(f"leg1_{joint}")
        front_r, rear_r = names.index(f"leg4_{joint}"), names.index(f"leg2_{joint}")
        error = error + torch.square(dev[:, front_l] - dev[:, rear_l])
        error = error + torch.square(dev[:, front_r] - dev[:, rear_r])
    return error


def heading_velocity_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min_speed: float = 0.15
) -> torch.Tensor:
    """Penalize the base_link for not facing the direction it is actually traveling in (i.e. veering/drift).

    Purely self-referential (heading vs. the robot's own measured velocity direction) -- NOT the same
    mechanism as move_to_target_bonus/base_heading_proj, which compare heading against a FIXED WORLD TARGET
    using the hardcoded LOCAL +X FORWARD_VEC_B (isaaclab_physx, not configurable). That mechanism is
    structurally broken for this robot (v1x1's legs propel along local Y, not X) and was already ruled out
    as the cause of the observed left-veer: the from-scratch side-test lineage trained with move_to_target
    at weight 0.0 for its ENTIRE history and still veers, so the fixed-target heading term can't be the
    cause here. This term instead uses this robot's own correct anatomical forward axis, LOCAL +Y (front
    legs mount on local +Y, rear on local -Y, per the source xacro), and simply asks "is the chassis facing
    the direction it's actually moving," independent of any world target -- can't hit the same structural
    bug since it isn't built on FORWARD_VEC_B at all. Only active above min_speed, since heading error is
    undefined/noisy while nearly stationary (e.g. right after a reset).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    forward_local = torch.zeros_like(asset.data.root_quat_w.torch[:, :3])
    forward_local[:, 1] = 1.0
    forward_world = quat_apply(asset.data.root_quat_w.torch, forward_local)[:, :2]
    forward_world = forward_world / (torch.linalg.norm(forward_world, dim=-1, keepdim=True) + 1e-6)

    vel_xy = asset.data.root_lin_vel_w.torch[:, :2]
    speed = torch.linalg.norm(vel_xy, dim=-1)
    vel_dir = vel_xy / (speed.unsqueeze(-1) + 1e-6)

    misalignment = torch.sum(torch.square(forward_world - vel_dir), dim=-1)
    return torch.where(speed > min_speed, misalignment, torch.zeros_like(misalignment))


def velocity_direction_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_command_speed: float = 0.1,
) -> torch.Tensor:
    """Reward the robot's ACTUAL body-frame velocity direction for pointing the same way as the COMMANDED
    velocity direction -- a direct, task-space "are you actually heading toward the target" signal.

    NOT the same as heading_velocity_l2 (this file, above), which compares BODY ORIENTATION to the robot's
    own velocity direction -- that term was correctly dropped for the omni-directional task (v1x1 is
    symmetric across all 4 legs, so it shouldn't need to turn to face its direction of travel; forcing that
    would fight legitimate sideways/backward commands). This term instead compares two VELOCITY vectors
    (actual vs. commanded), never touching body orientation at all -- fully compatible with strafing/
    backward motion, since a v1x1 crab-walking sideways with its body still facing "forward" gets full credit
    here as long as its actual travel direction matches the commanded one.

    Added because runs 2 and 3 both showed reward/trot_gait/episode-length climbing while the robot visibly
    kept "wiggling in place" rather than translating toward the commanded direction (confirmed live via the
    command debug-vis arrows: green command arrow relatively fixed, blue actual-velocity arrow spinning
    without settling toward it) -- track_lin_vel_xy_exp's combined magnitude+direction L2 error gives partial
    credit for a lot of near-miss states, apparently not enough on its own to force the policy to commit to a
    clear direction. Cosine similarity (dot product of unit vectors, range [-1, 1]) isolates the DIRECTION
    part of that problem specifically, rewarded independent of how well the magnitude happens to match.
    Gated on commanded speed (not actual speed, unlike heading_velocity_l2's gating) since alignment is
    undefined for a near-zero command and we want this active as soon as a real direction is commanded, even
    before the robot has learned to move at all.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    cmd_xy = command[:, :2]
    cmd_speed = torch.linalg.norm(cmd_xy, dim=-1)
    cmd_dir = cmd_xy / (cmd_speed.unsqueeze(-1) + 1e-6)

    actual_xy = asset.data.root_lin_vel_b.torch[:, :2]
    actual_speed = torch.linalg.norm(actual_xy, dim=-1)
    actual_dir = actual_xy / (actual_speed.unsqueeze(-1) + 1e-6)

    alignment = torch.sum(cmd_dir * actual_dir, dim=-1)
    return torch.where(cmd_speed > min_command_speed, alignment, torch.zeros_like(alignment))


def tibia_vertical_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the tibia (joint4) links for not pointing straight down.

    Each tibia link's own local -Z axis runs along its length toward the foot (confirmed from the source
    URDF: the now-pruned end-effector marker sat at local (0, 0, -0.242) in the tibia's own frame). This
    rotates that local axis into world frame per tibia and penalizes its XY component -- zero when the
    tibia hangs perfectly vertical, growing as it tilts sideways. Same principle as the core
    isaaclab.envs.mdp.rewards.flat_orientation_l2 (which penalizes projected-gravity XY on the root body),
    just generalized to an arbitrary body's own local axis since that function is root-only.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]  # (N, num_tibias, 4)
    down = torch.zeros_like(quat[..., :3])
    down[..., 2] = -1.0
    world_dir = quat_apply(quat, down)  # (N, num_tibias, 3)
    return torch.sum(torch.square(world_dir[..., :2]), dim=(-2, -1))


def foot_air_time_exceeded(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_air_time: float) -> torch.Tensor:
    """Terminate if any foot has been continuously airborne for longer than max_air_time.

    Catches "raise a leg and just hold it up" -- a degenerate policy that the feet_air_time reward alone
    doesn't punish (that reward only pays out the moment a foot lands, based on the *preceding* air time; a
    foot that never lands never triggers it, but also never gets penalized for staying up indefinitely).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    return torch.any(current_air_time > max_air_time, dim=1)


def foot_contact_time_exceeded(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_contact_time: float) -> torch.Tensor:
    """Terminate if any single foot has been continuously planted for longer than max_contact_time.

    Catches a policy that uses one foot as a fixed pivot/anchor instead of cycling all four through a gait.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    return torch.any(current_contact_time > max_contact_time, dim=1)


def all_feet_grounded_too_long(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_time: float) -> torch.Tensor:
    """Terminate if all four feet have been simultaneously, continuously in contact for longer than max_time.

    Catches "stand still in a wide/T-pose stance and don't move" -- the minimum current_contact_time across
    all four feet is exactly "how long has *every* foot in this set been continuously grounded" (if any one
    foot lifted more recently than the others, or is currently airborne, the min reflects that).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    return torch.min(current_contact_time, dim=1)[0] > max_time


def feet_air_time_unconditional(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Reward feet air time so the policy has an incentive to actually take steps, not just stand still.

    This is adapted from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards.feet_air_time, with the
    command-manager gating removed: that version only rewards air time while a nonzero velocity command is
    active (correct for velocity-tracking tasks where standing still is sometimes the right behavior). This
    task has no command manager -- it always wants forward progress toward a fixed target, so standing
    perfectly still is never the goal, and the reward should always be active.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    return torch.sum((last_air_time - threshold) * first_contact, dim=1)


class TrotGaitReward(_SpotGaitReward):
    """Trot-gait shaping reward: rewards each diagonal foot pair (e.g. front-left+rear-right) stepping in
    sync with each other, and out of sync with the other diagonal pair -- the classic quadruped trot pattern.

    Reused from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp.rewards.GaitReward (the
    sync/async foot-timing math is identical, no reason to reimplement it) with only the gating changed:
    the original only applies while a velocity *command* is nonzero, which this task has no command manager
    for. Gating on the robot's own measured body velocity instead achieves the same intent (don't demand a
    gait pattern while it's still just trying to stand up) without needing a command manager at all.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b.torch[:, :2], dim=1)
        return torch.where(body_vel > self.velocity_threshold, sync_reward * async_reward, 0.0)


class HeadingHoldBonus(ManagerTermBase):
    """Rewards the robot for maintaining the ORIGINAL world-frame direction a command asked for, over time --
    a genuine restoring force against the "slowly drifting/spinning in circles" failure mode (see runs 6-8),
    which `track_ang_vel_z_exp` and `velocity_direction_alignment` cannot provide on their own.

    Both of those existing terms only ever compare INSTANTANEOUS values each step: track_ang_vel_z_exp asks
    "is your turn rate close to the commanded turn rate right now," and velocity_direction_alignment asks "is
    your velocity direction close to the command's direction AS INTERPRETED IN YOUR CURRENT BODY FRAME right
    now." Neither has any memory of where the robot was originally pointed. Since the command is body-frame-
    relative, if the robot's own heading drifts even slightly (any nonzero residual yaw-rate error), the
    "target direction" those two terms compare against silently rotates along WITH the robot -- like a car
    with very slightly turned wheels holding constant speed: however precisely it holds that speed, a nonzero
    steering angle still traces a circle, not a straight line, because nothing is pulling it back toward its
    original heading. Tightening track_ang_vel_z_exp's std (tried in runs 7/8) only shrinks the RATE of that
    drift, it can't eliminate it structurally.

    This term fixes that by capturing a REFERENCE direction in WORLD frame -- not body frame -- at the moment
    each command is issued (detected by the raw command tensor changing since the last step), and rewarding
    the robot's actual world-frame velocity direction for staying close to THAT fixed reference for the rest
    of the command's duration, regardless of how the robot's own body orientation drifts in the meantime.
    Complementary to (not a replacement for) the existing tracking rewards -- this only addresses direction
    HOLDING over time, not speed/magnitude tracking.

    IMPORTANT GATE (added after run 11's postmortem): only applies when the COMMANDED ang_vel_z is near zero
    (|command| <= max_ang_vel_for_hold). For a command that also asks for real rotation, the robot's world-
    frame travel direction is SUPPOSED to curve over time as the turn takes effect -- that's correct
    execution, not drift. Since the command tensor (including ang_vel_z) is held exactly constant for the
    whole resample window, checking the CURRENT command's ang_vel_z each step correctly disables this reward
    for the entire window whenever a real turn was commanded, re-enabling automatically once a new,
    near-straight command begins.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._prev_command = torch.zeros(self.num_envs, 3, device=self.device)
        self._reference_world_dir = torch.zeros(self.num_envs, 2, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        min_command_speed: float = 0.1,
        max_ang_vel_for_hold: float = 0.2,
    ) -> torch.Tensor:
        command = env.command_manager.get_command(command_name)
        # a resample changes the raw command tensor's VALUE (held exactly constant between resamples by
        # UniformVelocityCommand), so exact equality reliably detects "a new command just started" -- also
        # naturally covers the very first step after any reset, since _initialized starts False there.
        changed = torch.any(command != self._prev_command, dim=-1) | ~self._initialized
        if torch.any(changed):
            cmd_xy = command[changed, :2]
            quat = self.asset.data.root_quat_w.torch[changed]
            cmd_xyz = torch.zeros(cmd_xy.shape[0], 3, device=self.device)
            cmd_xyz[:, :2] = cmd_xy
            world_dir = quat_apply(quat, cmd_xyz)[:, :2]
            self._reference_world_dir[changed] = world_dir
            self._initialized[changed] = True
        self._prev_command = command.clone()

        ref_speed = torch.linalg.norm(self._reference_world_dir, dim=-1)
        ref_dir = self._reference_world_dir / (ref_speed.unsqueeze(-1) + 1e-6)

        actual_xy = self.asset.data.root_lin_vel_w.torch[:, :2]
        actual_speed = torch.linalg.norm(actual_xy, dim=-1)
        actual_dir = actual_xy / (actual_speed.unsqueeze(-1) + 1e-6)

        alignment = torch.sum(ref_dir * actual_dir, dim=-1)
        straight_command = torch.abs(command[:, 2]) <= max_ang_vel_for_hold
        active = (ref_speed > min_command_speed) & straight_command
        return torch.where(active, alignment, torch.zeros_like(alignment))


class HeadingDeviationExceeded(ManagerTermBase):
    """Terminate if the robot's actual world-frame velocity direction has deviated from its reference
    direction by more than `max_angle_deg`, CONTINUOUSLY, for longer than `sustained_duration` seconds.

    A hard backstop complementing HeadingHoldBonus's soft reward -- ending the episode outright (losing ALL
    remaining reward) is a much stronger deterrent against persistent circling than any reward weight, no
    matter how large. Uses the identical world-frame reference-capture logic as HeadingHoldBonus (see that
    class's docstring for the full "car with slightly turned wheels" reasoning for why a WORLD-frame
    reference, captured once per command and held fixed, is needed at all) -- but maintains its own
    independent state, since the termination manager and reward manager don't share term instances.

    `sustained_duration` (not instantaneous) is essential: v1x1's gait naturally has some per-stride wobble
    in its instantaneous velocity direction, and this should only catch PERSISTENT drift, not normal gait
    noise. `grace_period` gives the robot a fair beat to physically reorient toward a brand-new command
    before deviation starts being counted at all -- without it, the natural settling transient right after
    every resample (now every 5s, per the shortened resampling_time_range) could trigger false terminations
    for doing exactly the right thing (reacting to a direction change), not for actually circling.

    IMPORTANT GATE (added after run 11's postmortem): only applies when the COMMANDED ang_vel_z is near zero
    (|command| <= max_ang_vel_for_hold) -- same reasoning as HeadingHoldBonus's identical gate. Without this,
    run 11 pinned at ~100% termination rate and never recovered across 450+ iterations: a meaningful fraction
    of commands include real ang_vel_z, for which a genuinely curving travel direction is CORRECT execution,
    not drift, and this termination was punishing both identically.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._prev_command = torch.zeros(self.num_envs, 3, device=self.device)
        self._reference_world_dir = torch.zeros(self.num_envs, 2, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._time_since_resample = torch.zeros(self.num_envs, device=self.device)
        self._deviation_duration = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._initialized[env_ids] = False
        self._time_since_resample[env_ids] = 0.0
        self._deviation_duration[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        max_angle_deg: float = 30.0,
        sustained_duration: float = 1.0,
        grace_period: float = 1.0,
        min_command_speed: float = 0.1,
        max_ang_vel_for_hold: float = 0.2,
    ) -> torch.Tensor:
        command = env.command_manager.get_command(command_name)
        changed = torch.any(command != self._prev_command, dim=-1) | ~self._initialized
        if torch.any(changed):
            cmd_xy = command[changed, :2]
            quat = self.asset.data.root_quat_w.torch[changed]
            cmd_xyz = torch.zeros(cmd_xy.shape[0], 3, device=self.device)
            cmd_xyz[:, :2] = cmd_xy
            world_dir = quat_apply(quat, cmd_xyz)[:, :2]
            self._reference_world_dir[changed] = world_dir
            self._initialized[changed] = True
            self._time_since_resample[changed] = 0.0
            self._deviation_duration[changed] = 0.0
        self._prev_command = command.clone()
        self._time_since_resample += env.step_dt

        ref_speed = torch.linalg.norm(self._reference_world_dir, dim=-1)
        ref_dir = self._reference_world_dir / (ref_speed.unsqueeze(-1) + 1e-6)

        actual_xy = self.asset.data.root_lin_vel_w.torch[:, :2]
        actual_speed = torch.linalg.norm(actual_xy, dim=-1)
        actual_dir = actual_xy / (actual_speed.unsqueeze(-1) + 1e-6)

        cos_angle = torch.clamp(torch.sum(ref_dir * actual_dir, dim=-1), -1.0, 1.0)
        angle_deg = torch.rad2deg(torch.acos(cos_angle))

        past_grace = self._time_since_resample > grace_period
        command_active = ref_speed > min_command_speed
        straight_command = torch.abs(command[:, 2]) <= max_ang_vel_for_hold
        exceeding = (angle_deg > max_angle_deg) & past_grace & command_active & straight_command

        self._deviation_duration = torch.where(
            exceeding, self._deviation_duration + env.step_dt, torch.zeros_like(self._deviation_duration)
        )

        return self._deviation_duration > sustained_duration

##
# NOTE: the color-coded command-arrow visualization (V1X1UniformVelocityCommand/TURN_ARROW_MARKER_CFG) that
# lived here was REMOVED on 2026-09-30 after it was identified as the cause of a persistent USD/Kit startup
# crash (in libusd_tf.so's TfRegistryManager, ~2s into every launch) that survived both a reboot and clearing
# shader/derived-data caches. A stock Ant task launched fine, isolating the problem to this module's new code;
# removing it (not just its usage in v1x1_velocity_env_cfg.py -- the module-level VisualizationMarkersCfg
# construction ran on import regardless) resolved it. If revisiting this idea, investigate WHY constructing a
# second VisualizationMarkersCfg prototype set at module import time crashes USD's type registry -- possibly
# needs to be constructed lazily (inside a function/class __init__) rather than as a bare module-level global.
##
