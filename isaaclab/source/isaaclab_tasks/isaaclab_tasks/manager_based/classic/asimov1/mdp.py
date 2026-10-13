# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local MDP terms for the asimov-1 stand-and-balance task.

Almost everything this task needs is pulled directly from the generic `isaaclab.envs.mdp` package (see
`asimov1_env_cfg.py`) -- asimov-1's conventional bilateral `left_*`/`right_*` joint naming is fully
compatible with those regex-driven generic terms, unlike v1x1's local-+Y-forward quirk which needed a large
bespoke mdp module. The only things genuinely missing from the stock library are these two: a direct
"don't drift" penalty on base velocity. Stock `lin_vel_z_l2`/`ang_vel_xy_l2` penalize the velocity
components that don't matter for *walking* tasks (vertical bounce, roll/pitch rate) -- but this is a
*standing-still* task, so the components that DO matter (horizontal drift, any yaw/rotation at all) have no
stock equivalent, since no walking task wants those penalized to zero.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def lin_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize horizontal base drift -- any nonzero xy velocity -- using an L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b.torch[:, :2]), dim=1)


def ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize any base rotation (all 3 axes) -- this task has no commanded turn, unlike a walking task."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b.torch), dim=1)


def body_vertical_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_tilt: float | None = None) -> torch.Tensor:
    """Penalize the given body link(s) tilting away from vertical in the world frame. Same mechanism as the
    stock `flat_orientation_l2` root-body check (project world-down gravity into the body's own frame,
    penalize the horizontal/XY components), just applied to arbitrary non-root bodies via `asset_cfg`.
    Reused for two separate reward terms in this task: shin verticality (left/right_knee_link -- the segment
    between knee and ankle in this URDF's naming), added live mid-session because the policy was observed
    persistently kicking one leg out to the side; and torso verticality (waist_yaw_link, the chest segment
    above the pelvis), added right after since `flat_orientation_l2` only checks the pelvis/root, not this
    separate upper-body segment, which could tilt independently. See asimov1_humanoid_project.md memory for
    the session both were added in.

    `max_tilt` (optional, default None = unconstrained, exactly the original standing-task behavior) creates
    a dead-zone: only tilt BEYOND this value is penalized, per body, before summing. Added for the velocity/
    AMP task's shin_vertical term after measuring real shin tilt in the AMP reference clip
    (policy_delay_walk_slow.npz): swing-phase shin tilt reaches a 95th percentile of 0.77 and an absolute
    max of ~0.82 (vs. this function's own torso measurement on the SAME clip of only ~0.13 max -- the torso
    stays genuinely near-vertical during real walking, but the shin does not, by necessity of swinging a
    bent knee forward). Without this dead-zone, even the reduced -0.3 weight was a non-trivial quadratic
    penalty exactly when attempting a big, confident swing, directly fighting the stride amplitude other
    terms (foot_clearance, foot_stuck, foot_swing_too_short) were trying to encourage.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat_w = asset.data.body_quat_w.torch[:, asset_cfg.body_ids, :]
    num_envs, num_bodies = body_quat_w.shape[0], body_quat_w.shape[1]
    gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=body_quat_w.device).expand(num_envs, num_bodies, 3)
    projected_gravity_b = quat_apply_inverse(body_quat_w.reshape(-1, 4), gravity_vec_w.reshape(-1, 3))
    projected_gravity_b = projected_gravity_b.reshape(num_envs, num_bodies, 3)
    per_body_tilt_sq = torch.sum(torch.square(projected_gravity_b[..., :2]), dim=2)
    if max_tilt is not None:
        per_body_tilt_sq = torch.clamp(per_body_tilt_sq - max_tilt**2, min=0.0)
    return torch.sum(per_body_tilt_sq, dim=1)


def whole_body_angular_momentum_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize whole-body angular momentum about the robot's own (mass-weighted) center of mass, L2 squared.

    Added per Menlo's own published reward docs for asimov-1's real locomotion stack: they add this term
    (alongside `body_ang_vel`) specifically because "Asimov has a narrower stance than many humanoid
    baselines... increases lateral balance sensitivity" -- motivated as a lateral-stability penalty for a
    narrow-stance biped, not a generic smoothness term. Added live mid-session when `hip_yaw_limit` was
    observed climbing across the omni-directional command-range curriculum widening (2.7%->5.1%->5.2%->
    12.3% as lateral/backward commands were introduced) -- a specific, mechanistically-motivated candidate
    for that exact symptom, not a shot in the dark.

    Computed as the ORBITAL (translational) contribution only -- `sum_i(m_i * cross(r_i - r_com, v_i))`,
    where `r_com` is the mass-weighted center of mass across all bodies -- omitting each body's own spin
    contribution (`I_i @ omega_i`, which would need per-body inertia tensors). This is a standard
    simplification for multi-link systems: the orbital term (limbs swinging relative to the body's own CoM)
    dominates total angular momentum for a humanoid far more than each individual link's own spin about its
    own axis. Weight -0.03 taken directly from Menlo's own published value for this exact term, not guessed.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_mass = asset.data.body_mass.torch  # (N, B)
    body_pos = asset.data.body_com_pos_w.torch  # (N, B, 3)
    body_vel = asset.data.body_com_lin_vel_w.torch  # (N, B, 3)
    total_mass = torch.sum(body_mass, dim=1, keepdim=True)  # (N, 1)
    com = torch.sum(body_mass.unsqueeze(-1) * body_pos, dim=1) / total_mass  # (N, 3)
    r = body_pos - com.unsqueeze(1)  # (N, B, 3)
    angular_momentum = torch.sum(body_mass.unsqueeze(-1) * torch.cross(r, body_vel, dim=-1), dim=1)  # (N, 3)
    return torch.sum(torch.square(angular_momentum), dim=1)


def leg_pose_tolerance_l1(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, tolerances: dict[str, float]
) -> torch.Tensor:
    """Penalize leg joint deviation from the default pose, but only BEYOND a per-joint-type tolerance --
    the same dead-zone discipline already used by `body_vertical_l2`'s `max_tilt` (penalize the real extreme,
    not the whole range), applied here to joint ANGLES instead of body tilt.

    Added per Menlo's own published reward docs for asimov-1's real locomotion stack: they use "asymmetric
    pose tolerances" per joint (NOT a uniform tolerance) specifically because "uniform pose tolerances across
    all joints are not appropriate for Asimov... the legs use different tolerances depending on the joint and
    the hardware structure." Their representative values (hip_pitch=0.5, hip_roll=0.25, hip_yaw=0.2,
    knee=0.5, ankle_pitch=0.2, ankle_roll=0.12) are used directly here via the `tolerances` dict (keyed by
    the SAME regex substring convention as `asset_cfg.joint_names`, matched by substring against each
    resolved joint's name) -- not guessed, their own stated values. A plain `joint_deviation_l1` (zero
    tolerance) would fight normal walking excursions directly, the same problem already solved for shin/torso
    tilt elsewhere in this file -- this term only costs something once a joint moves MORE than its own
    tolerance, not for any excursion at all.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    deviation = torch.abs(joint_pos - default_joint_pos)
    joint_names = [asset.data.joint_names[i] for i in asset_cfg.joint_ids]
    tol = torch.zeros(len(joint_names), device=deviation.device)
    for i, name in enumerate(joint_names):
        for pattern, value in tolerances.items():
            if pattern in name:
                tol[i] = value
                break
    excess = torch.clamp(deviation - tol.unsqueeze(0), min=0.0)
    return torch.sum(excess, dim=1)


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, ratio_threshold: float = 4.0) -> torch.Tensor:
    """Penalize a foot contact whose horizontal force is large relative to its vertical force -- a foot
    being DRAGGED sideways rather than loaded cleanly underfoot, the standard legged-robot "stumble" signal
    (same formula family used across the legged_gym/ANYmal-style RL literature: horizontal force exceeding
    some multiple of vertical force flags an unstable foot strike). Added per Menlo's own published reward
    docs (`feet_stumble`, weight -1.25, "discourage unstable foot strikes") -- no stock Isaac Lab equivalent
    exists, this is a genuinely new term, not a reused one.

    `ratio_threshold=4.0` is the standard default from that literature, not re-derived for asimov-1
    specifically -- flagged as a placeholder like any other un-measured numeric default in this project,
    worth revisiting if stumble detection looks too trigger-happy or too lax once watched live.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :]  # (N, feet, 3)
    horizontal = torch.norm(forces[..., :2], dim=-1)
    vertical = torch.abs(forces[..., 2])
    stumbling = horizontal > ratio_threshold * vertical
    return torch.sum(stumbling.float(), dim=1)


def feet_contact_force_limit(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_force: float = 350.0
) -> torch.Tensor:
    """Penalize ground reaction force magnitude above `max_force` per foot -- discourages aggressive
    stomping and protects a real robot from unnecessary impact loading. Added per Menlo's own published
    reward docs (`feet_contact_force_limit`, weight -5e-4, "penalizes forces above approximately 350N") --
    their exact threshold, not guessed.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :]  # (N, feet, 3)
    force_mag = torch.norm(forces, dim=-1)
    excess = torch.clamp(force_mag - max_force, min=0.0)
    return torch.sum(excess, dim=1)


def foot_airborne_exceeded(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_air_time: float) -> torch.Tensor:
    """Terminate if either foot has been continuously airborne longer than max_air_time.

    This is a standing-balance task (unlike v1x1's walking task, which rewards feet cycling through regular
    air time) -- any sustained foot-lift here signals falling/hopping, not a valid balance strategy, so it's
    always bad and always worth cutting off. Same `current_air_time` ContactSensor mechanism v1x1 uses for
    its own `foot_air_time_exceeded`, just inverted intent (there it catches "holding one leg up too long
    during a gait"; here it catches "lost ground contact at all while just trying to stand").
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    return torch.any(current_air_time > max_air_time, dim=1)


def foot_stuck_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    max_contact_time: float = 1.0,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize a foot that's stayed planted far longer than a normal stance phase while actually commanded
    to move -- added for the velocity/AMP task after the user observed good balance but no net translation:
    `feet_air_time_positive_biped` only rewards GOOD alternation when it happens, nothing in this task
    previously penalized the absence of it, so "balance well by barely shifting weight and never really
    stepping" was a viable, unpunished strategy.

    `max_contact_time=1.0` is grounded in the AMP reference clip (policy_delay_walk_slow.npz): measured by
    thresholding each foot's world-Z height against its own near-ground minimum, real single-foot stance
    phases have median duration 0.62-0.64s (90th percentile ~0.68s) -- 1.0s sits well above normal stance
    variation, catching genuine stalling, not ordinary gait rhythm.

    Gated on command magnitude (`command_threshold=0.1`, same convention as `feet_air_time_positive_biped`/
    `stand_still_joint_deviation_l1` elsewhere in this task) so it never fights the rel_standing_envs
    fraction that's actually supposed to hold still.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    excess = torch.clamp(contact_time - max_contact_time, min=0.0)
    penalty = torch.sum(excess, dim=1)
    command = env.command_manager.get_command(command_name)
    gate = (torch.linalg.norm(command[:, :2], dim=1) > command_threshold).float()
    return penalty * gate


def foot_swing_too_short_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    min_swing_time: float = 0.25,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """The mirror image of `foot_stuck_penalty` -- penalizes a foot that lifted and landed again TOO
    QUICKLY, the direct signature of "tippy-tapping" (rapid, low-amplitude taps rather than committed
    strides). Added after the user independently floated this idea live ("penalize the steps being too
    fast... slow/smooth out the steps") and observed tippy-tapping persisting (reduced, not eliminated) even
    after `foot_stuck_penalty` + a higher track_lin_vel_xy weight both landed.

    `min_swing_time=0.25` is grounded in the AMP reference clip (policy_delay_walk_slow.npz): measured by
    thresholding each foot's world-Z height, real swing (air) phases have an observed minimum of 0.24-0.30s
    and a median of 0.41-0.42s -- 0.25s sits just below the real gait's own fastest steps, catching genuine
    tippy-tapping without fighting any step the reference gait itself actually takes.

    Only pays out at the exact moment a foot LANDS (`compute_first_contact`), scaled by how far the
    just-completed swing (`last_air_time`) fell short of the minimum -- same event-timing convention as the
    stock `feet_air_time` reward this mirrors (`(last_air_time - threshold) * first_contact`), just
    penalizing the opposite side of that same number. Gated on command magnitude, same convention as
    `foot_stuck_penalty`, so it never fights the rel_standing_envs fraction.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    shortfall = torch.clamp(min_swing_time - last_air_time, min=0.0) * first_contact.float()
    penalty = torch.sum(shortfall, dim=1)
    command = env.command_manager.get_command(command_name)
    gate = (torch.linalg.norm(command[:, :2], dim=1) > command_threshold).float()
    return penalty * gate


def feet_crossed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float = -0.02) -> torch.Tensor:
    """Terminate if the feet cross laterally -- left foot ends up on the right foot's side or vice versa.

    `asset_cfg.body_names` must list the LEFT foot first, then the RIGHT foot (order matters here, unlike
    most other terms in this file). Uses world-frame Y position -- confirmed from the URDF mount offsets
    that Y is this robot's left-right axis (left_* joints sit at +Y, right_* at -Y relative to the pelvis).
    Default threshold=-0.02 (a small negative margin, not a strict 0.0) tolerates feet standing close
    together/briefly touching without false-triggering on normal narrow-STANDING-stance noise -- this is
    the right default for the standing task this term originated in.

    `threshold` made an explicit parameter live mid-session once the velocity/AMP task needed a much looser
    value: measured directly from the AMP reference motion clip (policy_delay_walk_slow.npz), a real
    demonstrated walking gait's left_y-right_y separation legitimately swings as low as -0.309m as NORMAL
    gait dynamics (15.3% of the clip's own frames would trip the old -0.02 default) -- the standing-task
    default is actively wrong for anything that actually walks.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :]
    left_y = body_pos_w[:, 0, 1]
    right_y = body_pos_w[:, 1, 1]
    return (left_y - right_y) < threshold


def feet_too_wide(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_separation: float) -> torch.Tensor:
    """Terminate if the feet spread wider than `max_separation`. Added live mid-session after the
    feet-crossing fixes (termination_penalty, feet_separation) pushed the policy the OTHER direction --
    some envs started widening their stance well past a normal balance-correction width. Same
    [left, right]-ordered body convention as `feet_crossed`/`feet_separation_l2`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :]
    left_y = body_pos_w[:, 0, 1]
    right_y = body_pos_w[:, 1, 1]
    return (left_y - right_y) > max_separation


def feet_separation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_separation: float) -> torch.Tensor:
    """Continuously penalize the feet getting closer together than `min_separation`, as a smooth, graded
    signal -- added because `feet_crossed` alone (a binary termination) gives zero gradient as the legs
    approach each other, only a cliff-edge penalty at the exact moment of crossing. The policy found and
    exploited this: even with `termination_penalty` active, it discovered that swinging the left leg inward
    until it crosses is still a cheap, repeatable way to end episodes early (confirmed live -- feet_crossed
    went back to firing on 100% of episodes after a brief suppression). This term pushes back the WHOLE
    approach, not just the final step, same body-order convention as `feet_crossed` (left foot first).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :]
    left_y = body_pos_w[:, 0, 1]
    right_y = body_pos_w[:, 1, 1]
    separation = left_y - right_y
    return torch.square(torch.clamp(min_separation - separation, min=0.0))


def feet_stance_width_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_separation: float) -> torch.Tensor:
    """Symmetric attractor toward `target_separation`, pulling back from EITHER side (too narrow OR too
    wide) -- unlike `feet_separation_l2`/`feet_too_wide`, which only guard the two extremes with a dead zone
    in between. `target_separation=0.215` is this robot's own natural resting stance width at the all-zero
    spawn pose (computed by summing the leg joint chain's Y-offsets at zero angle: hip_pitch -> hip_roll ->
    hip_yaw -> knee -> ankle_pitch -> ankle_roll, all rpy=0 so offsets add directly) -- added per explicit
    request to have the policy gravitate around its own natural idle stance rather than just avoiding the
    hard extremes. Same [left, right]-ordered body convention as the other feet_* terms in this file.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :]
    left_y = body_pos_w[:, 0, 1]
    right_y = body_pos_w[:, 1, 1]
    separation = left_y - right_y
    return torch.square(separation - target_separation)


class HeadingHoldBonus(ManagerTermBase):
    """Rewards holding the SAME world-frame yaw heading the robot started each episode with.

    Directly analogous to v1x1's own `HeadingHoldBonus` (added for its omni-directional walking task after
    discovering that velocity-only penalties have no restoring force against accumulated drift -- a small,
    steady yaw-rate error barely dents a per-step velocity penalty but still ends up facing a totally
    different direction after a long episode). asimov1 has no commanded direction to key off of (this is a
    standing task, not a locomotion task), so the reference moment is simply "right after each reset"
    instead of "whenever the command changes" -- otherwise the same mechanism: capture a REFERENCE yaw once
    per episode, reward staying close to THAT fixed reference over time, which `ang_vel_l2` (current rate
    only, no memory of where the episode started) cannot provide on its own.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._reference_yaw = torch.zeros(self.num_envs, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._initialized[env_ids] = False

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.3
    ) -> torch.Tensor:
        current_yaw = euler_xyz_from_quat(self.asset.data.root_quat_w.torch)[2]
        need_init = ~self._initialized
        if torch.any(need_init):
            self._reference_yaw[need_init] = current_yaw[need_init]
            self._initialized[need_init] = True
        yaw_error = wrap_to_pi(current_yaw - self._reference_yaw)
        return torch.exp(-torch.square(yaw_error) / std**2)


def recovery_step_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    tilt_threshold: float,
    tilt_range: float,
) -> torch.Tensor:
    """DIRECTIONAL version -- upgraded live per explicit request: "hint to them to kick their leg out
    toward the direction they are falling if they are falling." The original version of this function just
    rewarded ANY foot being airborne while leaning (confirmed engaging but non-specific); this rewards the
    airborne foot's WORLD-frame horizontal velocity for being aligned with the direction the robot is
    actually tipping toward -- a real "step toward the fall" signal, not just "a foot happens to be up."

    Lean direction is computed in WORLD frame as the horizontal component of the robot's own local up-axis
    rotated into world frame: if the robot tips forward, its up-vector leans forward too (for a body
    rotating about its support base), so stepping in THAT same direction is the physically correct catch.
    This sidesteps any body-frame sign-convention ambiguity -- gravity's world direction is a fixed
    constant, so reasoning in world frame throughout is simpler to verify than converting through the
    body's own rotation twice.

    Gated on tilt magnitude exactly as before (0 at `tilt_threshold`, full strength by
    `tilt_threshold + tilt_range`) -- `tilt_threshold=0.15`/`tilt_range=0.35` keeps the full-strength zone
    (~30 deg of tilt) well inside the `orientation` termination's hard cutoff (~57 deg), real early-warning
    room, not a last-ditch panic signal. Alignment (foot velocity dot lean direction) is clamped to >=0, so
    stepping AWAY from the lean, or just flailing a foot with no directional intent, earns nothing.

    REWARD-HACKING CAUTION (flagged explicitly per user request: "not reward hack towards falling and
    catching themselves"): this must never become an incentive to deliberately induce a lean just to farm
    the bonus. Mitigated structurally, not just by hand-waving: this reward's typical payout is a small
    fraction of a point (confirmed live in the previous, non-directional version -- peaked ~0.045 per
    episode-average with weight 2.5), while `flat_orientation` (-2.0 per step, continuously) and especially
    `termination_penalty` (-1500, one-time, if the induced lean actually tips into a fall) dwarf it. The
    expected value of deliberately destabilizing to chase this bonus is strongly negative as long as
    inducing a lean carries any real risk of actually falling, which it does -- there's no free lunch here,
    the risk/reward math is intentionally lopsided against gaming it.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    num_envs = asset.data.root_quat_w.torch.shape[0]
    up_local = torch.tensor([0.0, 0.0, 1.0], device=asset.data.root_quat_w.torch.device).expand(num_envs, 3)
    up_world = quat_apply(asset.data.root_quat_w.torch, up_local)
    tilt_xy = up_world[:, :2]
    tilt_mag = torch.norm(tilt_xy, dim=1)
    gate = torch.clamp((tilt_mag - tilt_threshold) / tilt_range, min=0.0, max=1.0)
    lean_dir = tilt_xy / (tilt_mag.unsqueeze(-1) + 1e-6)

    foot_vel_xy = asset.data.body_lin_vel_w.torch[:, sensor_cfg.body_ids, :2]
    current_air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    airborne = (current_air_time > 0.0).float()

    alignment = torch.sum(foot_vel_xy * lean_dir.unsqueeze(1), dim=-1)
    alignment = torch.clamp(alignment, min=0.0)
    directional_bonus = torch.sum(alignment * airborne, dim=1)

    return gate * directional_bonus


class InsufficientProgressTermination(ManagerTermBase):
    """Terminates if the base hasn't moved far enough NET distance over a fixed look-back window, given
    what it's actually been commanded to do. Added live mid-session per direct user suggestion, after
    persistent "bouncing/wiggling in place" was observed even with balance, foot-lift, and
    velocity-tracking rewards all engaging reasonably well: none of those measure NET progress over time,
    so oscillating back and forth (or micro-shuffling) can satisfy instantaneous reward some of the time
    without ever actually getting anywhere -- net displacement is the one thing that genuinely cancels that
    out.

    Unlike the feet-crossing exploit this project spent a long time fixing earlier in the standing task,
    there's no perverse incentive to trigger this one on purpose: doing so just means not performing the
    task at all, and termination_penalty (now -300 for this task, a real but no longer overwhelming
    deterrent -- see asimov1_velocity_env_cfg.py's RewardsCfg) still applies on top. This is a clean,
    well-defined failure condition, same category as feet_crossed/feet_too_wide/collapsed, not an ambiguous
    one that risks a new "fastest way to end this episode" dynamic.

    Mechanism: every `window_time` seconds, compares current horizontal position against a snapshot taken
    `window_time` seconds ago, then re-anchors the snapshot for the next window (a periodic checkpoint, not
    a full rolling buffer). Requires displacement over that window to be at least `min_speed_fraction` of
    what PERFECT command-tracking would have produced (scaled by the ACTUAL commanded speed at window-close
    time) -- so a near-zero command (e.g. the rel_standing_envs fraction) naturally requires near-zero
    displacement, never falsely triggering on legitimate standing.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        window_time = cfg.params["window_time"]
        self.window_steps = max(1, round(window_time / env.step_dt))
        self._ref_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._steps_since_ref = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._initialized[env_ids] = False
        self._steps_since_ref[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        window_time: float,
        min_speed_fraction: float = 0.3,
    ) -> torch.Tensor:
        current_pos = self.asset.data.root_pos_w.torch[:, :2]

        need_init = ~self._initialized
        if torch.any(need_init):
            self._ref_pos[need_init] = current_pos[need_init]
            self._steps_since_ref[need_init] = 0
            self._initialized[need_init] = True

        self._steps_since_ref += 1
        window_closed = self._steps_since_ref >= self.window_steps

        # FIXED live mid-session -- the original version measured displacement MAGNITUDE only
        # (torch.norm), direction-agnostic, so drifting backward at full speed when commanded FORWARD
        # satisfied this termination exactly as well as actually moving forward. Confirmed directly from
        # the user's own live observation ("even when commanded forward they tend to drift back") --
        # this termination had ZERO ability to catch that, since "moved a lot" was all it ever checked,
        # never "moved the right way." Now projects the displacement onto the COMMANDED direction (rotated
        # from the command's own yaw-aligned frame into world frame via the robot's current yaw, since
        # root_pos_w is a world-frame position) and requires that SIGNED projection to clear the threshold
        # -- moving backward when told to go forward now gives a NEGATIVE projection, correctly failing.
        displacement_vec = current_pos - self._ref_pos
        command = env.command_manager.get_command(command_name)
        command_xy_yaw_frame = command[:, :2]
        command_speed = torch.norm(command_xy_yaw_frame, dim=1)
        command_xyz_yaw_frame = torch.cat([command_xy_yaw_frame, torch.zeros_like(command_speed).unsqueeze(-1)], dim=-1)
        command_xyz_world = quat_apply(yaw_quat(self.asset.data.root_quat_w.torch), command_xyz_yaw_frame)
        command_dir_world = command_xyz_world[:, :2] / (command_speed.unsqueeze(-1) + 1e-6)
        projected_displacement = torch.sum(displacement_vec * command_dir_world, dim=1)
        min_displacement = command_speed * window_time * min_speed_fraction
        insufficient = window_closed & (projected_displacement < min_displacement)

        if torch.any(window_closed):
            self._ref_pos[window_closed] = current_pos[window_closed]
            self._steps_since_ref[window_closed] = 0

        return insufficient
