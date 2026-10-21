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
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass
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


def feet_only_contact_bonus(
    env: ManagerBasedRLEnv,
    feet_sensor_cfg: SceneEntityCfg,
    pelvis_sensor_cfg: SceneEntityCfg,
    standing_height: float,
    transition_width: float = 0.15,
    contact_threshold: float = 20.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Rewards the specific CONTACT PATTERN of standing: BOTH feet registering real ground-contact force
    AND the pelvis registering NONE -- a direct signal for "properly supported only by the feet," distinct
    from every other reward in this task, which infers posture indirectly from height/orientation/joint
    angles. User's own idea after run 14 showed the whole population converged on sitting (visually
    confirmed via a live screenshot) despite excellent height/orientation metrics: height/orientation alone
    can't tell "sitting with a perfectly upright torso" apart from "standing," since both can satisfy
    those numerically -- an actual ground-contact PATTERN can.

    Uses this asset's real `contact_forces` sensor (not a proxy) -- but scoped to what it actually has
    PhysX contact-report coverage on, confirmed at this asset's one-time USD conversion: feet + pelvis
    only (see `CONTACT_SENSOR_AVAILABLE_BODIES`). A literal thigh/femur-contact check (the user's other
    suggested signal) isn't available without re-converting the asset's USD, a bigger lift not done this
    session -- pelvis-contact-ABSENCE is used here as the best available proxy for "not sitting/lying on
    the butt or thighs," since lying/sitting postures reliably load the pelvis against the ground while a
    genuine standing posture does not.

    **Height-gated after a verification check showed the raw contact pattern alone fires on ~70-83% of
    steps for a largely RANDOM policy**, even after raising `contact_threshold` 1.0 -> 20.0 (a real load-
    bearing requirement, not just incidental touch) barely moved it (83% -> 72%). Root cause: "feet
    touching with force, pelvis not" is satisfied by plenty of ordinary floor postures too -- e.g. lying
    prone with the hips arched slightly off the ground while splayed legs press into it. Same lesson
    OpenHorizon's own equivalent reward (`feet_contact_when_high`) already encodes: they always gate it on
    proximity to standing height (their `standing_gate`), never reward the raw contact pattern alone. Added
    the identical phase-gate mechanism this file already uses elsewhere (`knee_bend_phase_bonus`/
    `final_pose_phase_penalty`) -- ramps 0->1 as height approaches `standing_height`, so only a contact
    pattern achieved WHILE ALSO meaningfully elevated counts.
    """
    feet_sensor: ContactSensor = env.scene.sensors[feet_sensor_cfg.name]
    feet_force = feet_sensor.data.net_forces_w.torch[:, feet_sensor_cfg.body_ids, :]
    both_feet_contact = (torch.norm(feet_force, dim=-1) > contact_threshold).all(dim=1)

    pelvis_sensor: ContactSensor = env.scene.sensors[pelvis_sensor_cfg.name]
    pelvis_force = pelvis_sensor.data.net_forces_w.torch[:, pelvis_sensor_cfg.body_ids, :]
    pelvis_contact = (torch.norm(pelvis_force, dim=-1) > contact_threshold).any(dim=1)

    asset: Articulation = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w.torch[:, 2]
    phase = torch.clamp((height - (standing_height - transition_width)) / transition_width, min=0.0, max=1.0)

    return (both_feet_contact & ~pelvis_contact).float() * phase


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

    FIXED a second time, live mid-session (2026-10-20): the "commanded direction" used for the projection
    was computed from the robot's CURRENT heading (at window-CLOSE time), not a heading anchored to when
    the window started. For a straight-walking command this is a non-issue (heading barely changes over
    one window), but for any sustained ang_vel_z command -- exactly the behavior the turning-focused reward
    change this same session was trying to train -- the robot's heading (and so "forward") rotates
    throughout the window. The displacement vector traces a curved/arced path, while the projection
    direction only reflects the FINAL heading, creating a real mismatch that grows with how much the robot
    actually turned -- i.e. this measured "genuine committed turning" as *worse* progress than it really
    was, specifically. Confirmed empirically, not just by this math: after raising track_ang_vel_z's reward
    weight (which correctly got the policy attempting real turns for the first time), this termination's
    own logged rate climbed from ~0% to ~65-68% of episodes within a few hundred iterations in two separate
    runs -- i.e. the moment the policy started doing the thing we wanted, this bug started killing those
    episodes before it could get credit for it. Fixed by snapshotting the heading (`_ref_quat`) at the same
    instant as the position snapshot (`_ref_pos`), and projecting against THAT heading instead of the
    current one -- removes the shifting-reference-frame artifact while leaving the termination's actual
    purpose (catching genuine stalling/wiggling) untouched.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        window_time = cfg.params["window_time"]
        self.window_steps = max(1, round(window_time / env.step_dt))
        self._ref_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self._ref_quat = torch.zeros(self.num_envs, 4, device=self.device)
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
        min_command_speed: float = 0.15,
    ) -> torch.Tensor:
        current_pos = self.asset.data.root_pos_w.torch[:, :2]
        current_quat = self.asset.data.root_quat_w.torch

        need_init = ~self._initialized
        if torch.any(need_init):
            self._ref_pos[need_init] = current_pos[need_init]
            self._ref_quat[need_init] = current_quat[need_init]
            self._steps_since_ref[need_init] = 0
            self._initialized[need_init] = True

        self._steps_since_ref += 1
        window_closed = self._steps_since_ref >= self.window_steps

        # Projects displacement onto the COMMANDED direction (rotated from the command's own yaw-aligned
        # frame into world frame via the heading ANCHORED at window-start, self._ref_quat -- not the
        # robot's current heading, see the class docstring's second FIXED paragraph above) and requires
        # that SIGNED projection to clear the threshold -- moving backward when told to go forward still
        # gives a negative projection (that part of the original direction-aware fix is unchanged).
        displacement_vec = current_pos - self._ref_pos
        command = env.command_manager.get_command(command_name)
        command_xy_yaw_frame = command[:, :2]
        command_speed = torch.norm(command_xy_yaw_frame, dim=1)
        command_xyz_yaw_frame = torch.cat([command_xy_yaw_frame, torch.zeros_like(command_speed).unsqueeze(-1)], dim=-1)
        command_xyz_world = quat_apply(yaw_quat(self._ref_quat), command_xyz_yaw_frame)
        command_dir_world = command_xyz_world[:, :2] / (command_speed.unsqueeze(-1) + 1e-6)
        projected_displacement = torch.sum(displacement_vec * command_dir_world, dim=1)
        min_displacement = command_speed * window_time * min_speed_fraction
        # min_command_speed is a true DEAD-ZONE, not just the proportional min_displacement scaling above --
        # added live mid-session (2026-10-20) as a residual fix once the heading-snapshot bug (see docstring)
        # was already fixed but insufficient_progress stayed high (~72.6%) under TurnBiasedVelocityCommand's
        # TURN-mode episodes, whose lin_vel band is narrowed to (-0.1,0.1) but NOT pinned to exactly 0.0 (a
        # deliberate choice, for sampling diversity -- see TurnBiasedVelocityCommandCfg's own docstring).
        # Even a small nonzero commanded speed sets a small-but-real min_displacement bar that a genuinely
        # correct, nearly-stationary pivot turn can still fail to clear (real CoM sway/stepping noise during
        # an in-place turn easily dips below a few cm of net projected progress). Below min_command_speed,
        # skip the check entirely rather than just scaling it down -- a true "near-zero command needs truly
        # no progress requirement," not merely a small one. 0.15 m/s sits just above TURN mode's worst-case
        # band magnitude (sqrt(0.1^2+0.1^2)=0.141), so legitimate turn-focused episodes are fully exempt,
        # while LINEAR-mode episodes (full lin_vel range) almost always exceed it and stay fully checked.
        insufficient = window_closed & (projected_displacement < min_displacement) & (command_speed >= min_command_speed)

        if torch.any(window_closed):
            self._ref_pos[window_closed] = current_pos[window_closed]
            self._ref_quat[window_closed] = current_quat[window_closed]
            self._steps_since_ref[window_closed] = 0

        return insufficient


##
# Fall-recovery (getup) task terms
##


def _upright_factor(asset: Articulation) -> torch.Tensor:
    """`-gravity_z` clamped to [0, 1] -- 1.0 fully vertical, 0.0 at/past horizontal. Shared by every
    height-based reward below (see their own docstrings for why this is needed): pelvis height achieved
    while badly tilted (e.g. via head/shoulder leverage against the ground) should not pay full height
    credit. Same underlying quantity `apply_assist_force`'s own upright gate uses, just unparameterized
    (no `min_upright` threshold) since a REWARD should stay smooth/proportional rather than cliff-edged.
    """
    return torch.clamp(-asset.data.projected_gravity_b.torch[:, 2], min=0.0, max=1.0)


def height_tracking_exp(
    env: ManagerBasedRLEnv, target_height: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Exponential-kernel height-tracking reward, scaled by how upright the robot currently is. Stock
    `isaaclab.envs.mdp.base_height_l2` is a pure L2 penalty with no stock exponential-kernel equivalent --
    mirrors `track_lin_vel_xy_exp`'s exact shape instead, since a graded exp kernel (vs. a harsh L2
    penalty) is a better fit for a task where almost every initial state starts far from the target height.

    **The upright scaling was added after run 10's `kneel_height_tracking` (same structural shape, see
    `height_milestone_exp`) re-triggered the ORIGINAL A-pose exploit** -- confirmed live and in the logs:
    iteration 721->1036, `orientation_upright` collapsed 1.49->0.02 while `kneel_height_tracking` spiked
    0.20->1.84 (near max) -- the robot discovered that head/shoulder-leverage against the ground raises
    pelvis height fast, and NEITHER `height_tracking` nor `height_milestone_exp` ever checked orientation,
    so that leverage paid full height credit regardless of posture. This is the THIRD time the exact same
    root cause has bitten this task (runs 1-7's original A-pose; run 8's assist force subsidizing it;
    now `kneel_height_tracking` re-opening it) -- fixed structurally here instead of patching a 4th
    symptom: any height-based reward now scales by `_upright_factor`, so raising the pelvis while tilted
    over (not just while cleanly climbing) earns proportionally less, closing the loophole at its source
    rather than only in whichever specific mechanism last exploited it.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.square(asset.data.root_pos_w.torch[:, 2] - target_height)
    return torch.exp(-height_error / std**2) * _upright_factor(asset)


def height_milestone_exp(
    env: ManagerBasedRLEnv, target_height: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """One-sided variant of `height_tracking_exp`, for use with an INTERMEDIATE waypoint target instead of
    the final goal height. `height_tracking_exp`'s full two-sided exp kernel is a symmetric bell curve --
    correct when `target_height` IS the actual final objective (reward decreasing on either side of the
    target is exactly right there), but actively WRONG for a milestone the robot is expected to climb PAST:
    confirmed via a real verification run where a `height_tracking_exp`-based sit-height term saturated
    near its max while the real height_tracking term (target=standing height) stayed flat -- the sit-height
    term's own reward was mathematically decreasing as the robot continued climbing past it, fighting the
    actual goal the whole time, not just risking it.

    This version only penalizes being BELOW target_height (error clamped to >=0) -- reward ramps up to
    full credit approaching the target from below, then STAYS at full credit for any height AT OR ABOVE
    it, so there's no incentive to come back down and no penalty for continuing to climb past it.

    Also scaled by `_upright_factor` -- see `height_tracking_exp`'s own docstring for why (this is the
    function `kneel_height_tracking`/`sit_height_tracking` both route through, and `kneel_height_tracking`
    specifically is what re-triggered the head-leverage exploit in run 10).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.clamp(target_height - asset.data.root_pos_w.torch[:, 2], min=0.0)
    return torch.exp(-torch.square(height_error) / std**2) * _upright_factor(asset)


class StabilityHoldBonus(ManagerTermBase):
    """Rewards CONSECUTIVE steps where height and orientation both sit near the standing target.

    Direct code answer to a known failure mode in getup RL (HoST, arXiv:2502.08378): a policy that briefly
    pops upright for one frame then immediately collapses can still accrue reward from per-step height/
    orientation terms alone. The counter here resets to zero the instant stability breaks, so a real, HELD
    getup is the only thing that pays out -- popping up and falling back down earns ~nothing.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._consecutive_stable_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._consecutive_stable_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        target_height: float = 0.63,
        height_tolerance: float = 0.08,
        tilt_tolerance: float = 0.3,
        required_hold_steps: int = 25,
    ) -> torch.Tensor:
        num_envs = self.asset.data.root_quat_w.torch.shape[0]
        up_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(num_envs, 3)
        up_world = quat_apply(self.asset.data.root_quat_w.torch, up_local)
        tilt_mag = torch.norm(up_world[:, :2], dim=1)
        height = self.asset.data.root_pos_w.torch[:, 2]
        stable = (tilt_mag < tilt_tolerance) & (torch.abs(height - target_height) < height_tolerance)
        self._consecutive_stable_steps = torch.where(
            stable, self._consecutive_stable_steps + 1, torch.zeros_like(self._consecutive_stable_steps)
        )
        return torch.clamp(self._consecutive_stable_steps.float() / required_hold_steps, max=1.0)


class RecoveryFeetPlantedBonus(ManagerTermBase):
    """Rewards CONSECUTIVE steps where BOTH feet maintain continuous ground contact (no lift-off), once
    already near-standing -- a direct answer to the user's own live observation of visible "tippy-tapping"
    (micro foot-lift jitter) and overall "wiggling" quality right after a real phase-stitch handoff.

    Why neither existing mechanism already caught this: `recovery_pose_alignment_exp`'s aggregate 23-joint
    L2 distance is a CONTINUOUS position signal, and a brief foot-tap (lift+replant within a fraction of a
    second) barely moves a joint-angle sum averaged over an entire episode's worth of steps -- foot contact
    is fundamentally a DISCRETE on/off pattern, not well-represented by continuous position error.
    `feet_only_contact_bonus` checks the right SENSOR but the wrong THRESHOLD/purpose (20N, tuned to
    distinguish "bearing real standing weight" from "sitting on the pelvis," not "is the foot airborne at
    all") and has no memory across steps, so it can't distinguish sustained double-support from rapid
    tap-tap-tap (each individual tap-down moment could still register >20N transiently).

    Mirrors `StabilityHoldBonus`'s own consecutive-counter pattern (same file, same anti-exploit
    discipline): the counter resets to 0 the instant EITHER foot loses contact, so a policy that taps and
    replants gets no more credit than one that's still airborne -- only genuinely SUSTAINED double-support
    pays out, same "popping up then immediately losing it earns ~nothing" philosophy `StabilityHoldBonus`
    already established for height/orientation.

    GATED by height>`gate_height` (same threshold convention as `recovery_pose_alignment_exp`/
    `recovery_hip_symmetry_l2`) so this doesn't fight the early climb, where feet genuinely need to move/
    push off the ground -- only matters once already close to standing.

    `contact_threshold=1.0` matches `gates.STAND_CONTACT_THRESHOLD_N`/`AssistCurriculum`'s own
    feet-in-contact check (a light "touching at all" threshold, not `feet_only_contact_bonus`'s heavier
    20N "bearing real weight" one -- the right choice here, since even a lightly-grazing foot mid-tap would
    still count as "in contact" under the heavier threshold, missing exactly the jitter being targeted).
    `required_ticks=30` (~0.5s at this task's control rate) is an unmeasured placeholder for "settled, not
    tapping," flagged for tuning off this run's own logged magnitude like every other placeholder here.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._consecutive_planted = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._consecutive_planted[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        gate_height: float,
        contact_threshold: float = 1.0,
        required_ticks: int = 30,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_cfg.name]
        forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :]
        both_planted = (torch.norm(forces, dim=-1) > contact_threshold).all(dim=-1)
        self._consecutive_planted = torch.where(
            both_planted, self._consecutive_planted + 1, torch.zeros_like(self._consecutive_planted)
        )
        height_gate = (self.asset.data.root_pos_w.torch[:, 2] > gate_height).float()
        progress = torch.clamp(self._consecutive_planted.float() / required_ticks, max=1.0)
        return progress * height_gate


class SitDwellPenalty(ManagerTermBase):
    """Continuous (never hard-terminating) penalty that GROWS the longer the robot dwells anywhere below
    near-standing height without making real climbing progress -- covers lying on the back/front near the
    floor AND resting at the sitting/kneeling waypoints with one mechanism, per the user's explicit request
    after run 11 ("tighten up the penalty for sitting too long... or staying on their backs for too long").

    Tracks each env's height against a reference that only refreshes the instant real progress is detected
    (so genuine climbing resets the clock rather than accruing penalty). A `_dwell_steps` counter grows
    while (a) height is below `max_height` and (b) at least `grace_steps` have passed since the last reset
    (a brief settle window -- the robot just free-fell and needs a moment before any "stalling" judgment is
    fair) and (c) no real progress has happened since the last refresh; it resets to 0 the instant any of
    those breaks. The returned penalty ramps 0->max_penalty over `window_steps` of continuous stalling.

    Deliberately graded, not a hard termination -- `StalledRecoveryTermination`/`HeadGroundTermination`
    (both still in this file, never wired in) already proved a hard pass/fail progress check fires
    universally before any skill exists to pass it with, collapsing every episode to the same early length.
    A continuous penalty doesn't truncate the episode even if it fires near-universally early in training
    (same as `upper_body_ground_penalty`'s own precedent of being near-universal early on, which has never
    been a problem) -- it just shifts the reward baseline down uniformly without destroying the learning
    signal, a far milder failure mode if the parameters turn out too aggressive.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._dwell_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ref_height = self.asset.data.root_pos_w.torch[:, 2].clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._dwell_steps[env_ids] = 0
        self._ref_height[env_ids] = self.asset.data.root_pos_w.torch[env_ids, 2]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        max_height: float = 0.55,
        min_progress: float = 0.03,
        window_steps: int = 120,
        grace_steps: int = 60,
        max_penalty: float = 1.0,
    ) -> torch.Tensor:
        height = self.asset.data.root_pos_w.torch[:, 2]
        below_ceiling = height < max_height
        past_grace = env.episode_length_buf > grace_steps
        progressed = (height - self._ref_height) > min_progress

        self._ref_height = torch.where(progressed, height, self._ref_height)
        active = below_ceiling & past_grace
        self._dwell_steps = torch.where(
            progressed | ~active, torch.zeros_like(self._dwell_steps), self._dwell_steps + 1
        )
        dwell_fraction = torch.clamp(self._dwell_steps.float() / window_steps, max=1.0)
        return dwell_fraction * max_penalty * active.float()


class RecoveryPositionDriftPenalty(ManagerTermBase):
    """Penalizes net horizontal displacement from wherever the episode started (i.e. where the robot fell),
    as distinct from per-step foot-sliding (that's `loco_mdp.feet_slide`, reused separately in this task's
    RewardsCfg). Added per the user's own real-world framing: these robots will eventually do household
    chores in tight hallways, where "recovers successfully but ends up a meter away from where it fell" is
    a real functional problem, not a cosmetic one -- confirmed by live observation of the trained policy
    visibly "scooting" across the floor while getting up.

    Anchors to each env's own root XY position at episode start (right after EventCfg's own random-fall
    `reset_base` term applies, since ManagerTermBase.reset() always fires after the EventManager's own
    'reset' mode terms) and penalizes the CURRENT distance from that anchor every step. The anchor-then-
    diff pattern mirrors `InsufficientProgressTermination`'s own root_pos_w usage in the walking task's
    mdp.py -- no env_origin subtraction needed, since env_origins is a per-env CONSTANT offset that cancels
    out in any same-env difference.

    `tolerance` is a dead-zone before any penalty starts (same "measure a real range, don't punish normal
    behavior" discipline as `shin_vertical`/`torso_vertical`'s own max_tilt tolerances) -- no motion-capture
    reference exists for this task's getup motion (unlike the AMP walking task's reference clip), so this is
    a reasoned placeholder (roughly one footstep's length -- minor weight-shift repositioning during a real
    getup is normal; several footsteps of net travel is the actual "scooting" behavior being targeted),
    flagged for visual-tuning like every other unmeasured placeholder in this project.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._anchor_xy = self.asset.data.root_pos_w.torch[:, :2].clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._anchor_xy[env_ids] = self.asset.data.root_pos_w.torch[env_ids, :2]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        tolerance: float = 0.15,
    ) -> torch.Tensor:
        current_xy = self.asset.data.root_pos_w.torch[:, :2]
        drift = torch.norm(current_xy - self._anchor_xy, dim=1)
        return torch.square(torch.clamp(drift - tolerance, min=0.0))


def upper_body_ground_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float
) -> torch.Tensor:
    """Continuous penalty against the given bodies (head/shoulder region) being low -- discourages using
    that region as a ground prop, the mechanism that enables the "A-pose" local optimum (head/shoulders
    down + legs forming the other points of a stable tripod, confirmed live via training video).

    NOTE: there is no separate head rigid body on this asset -- the URDF's 2-DOF neck
    (`neck_yaw_link_fixed`/`neck_pitch_link_fixed`) is unactuated and both fixed joints got merged into
    `waist_yaw_link` during this asset's one-time URDF->USD conversion (`--merge-joints`), confirmed via a
    direct rigid-body probe (no "head"/"neck" link exists in the converted asset's body list at all). This
    is meant to be called with BOTH `waist_yaw_link` and both `*_shoulder_pitch_link` bodies, giving a much
    better approximation of "head/shoulders near the ground" than the torso alone.

    REPLACES an earlier `torso_near_ground_penalty` version that used a fixed `min_height` threshold
    (clamp-based, zero penalty once above it) -- a real run showed the logged penalty shrinking toward zero
    over training while the user kept visually observing persistent head-resting live, meaning the robot
    had found a resting configuration where tracked body height stayed just above that threshold without
    the behavior actually stopping. This version is threshold-free and fully graded instead: penalty scales
    continuously with how far below `target_height` (the robot's own standing-height estimate) the tracked
    bodies sit, with NO height at which the penalty bottoms out to exactly zero before actually standing --
    removes the whole class of "found a resting height just above the cutoff" exploit.

    Deliberately a continuous PENALTY, not a termination -- this task has no terminations besides time_out
    by design (see module/env-cfg docstrings): most random fall orientations land with these bodies near
    the ground immediately, so a termination here would end nearly every episode before the robot has any
    chance to react.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tracked_height = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2].mean(dim=1)
    return torch.clamp(target_height - tracked_height, min=0.0)


def knee_bend_phase_bonus(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, standing_height: float, transition_width: float = 0.15
) -> torch.Tensor:
    """Rewards bending the knees while still low/recovering, PENALIZES bent knees once nearly standing --
    a continuous phase gate keyed on current height, not a hard switch. User's own insight: knee flexion is
    the natural mechanical lever for pushing up off the ground, but the same bent-knee pose that helps
    during the climb becomes wasted effort/instability once actually upright, so the incentive needs to
    flip sign partway through the episode rather than being fixed.

    `phase` ramps linearly from 0 (reward bending) to 1 (penalize bending) as height rises through the
    `transition_width`-wide band just below `standing_height` -- smooth, not a step function, so there's no
    sudden incentive flip at one exact height that could itself become exploitable.

    FIXED after a real run showed the SAME class of reward-hacking as `limb_pushoff_bonus` -- by iteration
    299/300 this had grown to be the single LARGEST reward component (1.47) while height_tracking/
    stability_hold/torso_upright all stayed essentially flat: rewarding raw knee-deviation MAGNITUDE
    unconditionally whenever height is low has no requirement that the bending is actually helping, so the
    policy can farm it by just flexing the knees back and forth while going nowhere. The REWARD side (low
    height) is now additionally gated on positive CoM vertical velocity, same principle as
    `limb_pushoff_bonus` -- bending only pays while it coincides with actually rising. The PENALTY side
    (near standing) is deliberately left UNGATED -- bent knees are wasteful/unstable once upright
    regardless of instantaneous vertical velocity, there's no equivalent exploit risk to guard against
    there.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    knee_deviation = torch.sum(
        torch.abs(asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]),
        dim=1,
    )
    height = asset.data.root_pos_w.torch[:, 2]
    phase = torch.clamp((height - (standing_height - transition_width)) / transition_width, min=0.0, max=1.0)
    rising = torch.clamp(asset.data.root_lin_vel_w.torch[:, 2], min=0.0)
    reward_component = knee_deviation * (1.0 - phase) * rising
    penalty_component = knee_deviation * phase
    return reward_component - penalty_component


def limb_pushoff_bonus(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Rewards actively loading the given limbs (hands/knees, via `sensor_cfg`) WHILE the robot is actually
    rising -- the intended mechanism for "use your hands/knees to push off the ground."

    Uses `joint_wrench`'s per-body reaction-force VERTICAL (Z) COMPONENT ONLY, clamped to positive, as a
    proxy for "pushing down into the ground through this limb" -- this is joint reaction load, not a
    direct ground-contact-force reading (this asset's `contact_forces` sensor only has the PhysX
    contact-report API active on 3 bodies -- feet + pelvis -- confirmed via a direct crash when the
    fall-recovery task first tried to use it on the full 8-body set; knees/wrists/waist never had it
    enabled at this asset's one-time USD conversion).

    FIXED after a real run showed reward-hacking: the original version summed full 3D force MAGNITUDE,
    which any vigorous, uncoordinated thrashing (confirmed live -- "doing the worm") can trigger just as
    easily as a genuine push, since lateral/rotational loading from flailing counts exactly the same as a
    deliberate downward push. Restricting to the clamped vertical component only counts force in the
    specific direction that could mechanically explain pushing the body upward, making it much harder for
    non-vertical thrashing to trigger spuriously.

    GATED on positive CoM vertical velocity specifically so this can't be farmed by just resting weight on
    a limb while static -- only effort that COINCIDES WITH rising counts, matching the same "no reward for
    camping in a convenient pose" principle as `StabilityHoldBonus` above.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.force.torch[:, sensor_cfg.body_ids]
    vertical_force = torch.clamp(force[..., 2], min=0.0)
    total_effort = torch.sum(vertical_force, dim=1)
    asset: Articulation = env.scene[asset_cfg.name]
    rising = torch.clamp(asset.data.root_lin_vel_w.torch[:, 2], min=0.0)
    return total_effort * rising


def hand_pushoff_force_bonus(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_force: float = 30.0,
    max_force: float = 150.0,
) -> torch.Tensor:
    """Rewards a DECISIVE vertical push-off force through the hand end-effectors specifically, graded
    between `min_force` and `max_force` -- a threshold-shaped bonus, unlike `limb_pushoff_bonus`'s
    unconditional linear-in-force reward (which also covers knees, combined). Added per the user's own
    insight after run 13: robots are currently either sitting or doing a belly-down worm -- neither
    involves a deliberate hands-push phase bridging those two states ("get their feet underneath them").

    Uses the SAME two gates `limb_pushoff_bonus` already proved safe (vertical-force-component-only,
    positive rising CoM velocity) -- the exact fixes that closed the original thrashing exploit (run 3)
    apply equally to any new hand-force reward, not just the original term, so this doesn't reopen that
    risk. The `min_force`/`max_force` ramp (instead of rewarding any positive force, however small) is the
    new piece here -- credits a clearly decisive push, not token/incidental hand contact. Both thresholds
    are first-pass guesses (no measured per-hand push-force data for this asset) -- watch the logged
    magnitude on the verification check before trusting them on a live run, same discipline as every other
    placeholder number in this file.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.force.torch[:, sensor_cfg.body_ids]
    vertical_force = torch.clamp(force[..., 2], min=0.0)
    total_force = torch.sum(vertical_force, dim=1)
    asset: Articulation = env.scene[asset_cfg.name]
    rising = torch.clamp(asset.data.root_lin_vel_w.torch[:, 2], min=0.0)
    graded_force = torch.clamp((total_force - min_force) / (max_force - min_force), min=0.0, max=1.0)
    return graded_force * rising


class StalledRecoveryTermination(ManagerTermBase):
    """Terminates if the robot's height hasn't meaningfully increased over a fixed look-back window --
    mirrors `InsufficientProgressTermination`'s exact mechanism (walking task, net-displacement-over-a-
    window) but keyed on height instead of commanded-direction displacement.

    Catches genuine stalling (e.g. resting in an A-pose, or settling into some OTHER equally-comfortable
    local optimum once the A-pose itself is discouraged by `torso_near_ground_penalty` above) without
    firing on legitimate pauses: the window is a deliberately generous fixed duration (not a tight
    per-step check), so briefly holding still mid-maneuver (e.g. repositioning hands/knees before a push)
    doesn't trip it -- only genuinely going nowhere for the FULL window does.

    Like `InsufficientProgressTermination`, there's no perverse incentive to trigger this on purpose --
    doing so just ends the episode without achieving anything, and the episode's own reward terms (height_
    tracking, stability_hold, etc.) already reward the alternative (making progress) far more than this
    termination's absence would ever be worth.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        window_time = cfg.params["window_time"]
        self.window_steps = max(1, round(window_time / env.step_dt))
        self._ref_height = torch.zeros(self.num_envs, device=self.device)
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
        window_time: float,
        min_height_gain: float = 0.05,
    ) -> torch.Tensor:
        current_height = self.asset.data.root_pos_w.torch[:, 2]

        need_init = ~self._initialized
        if torch.any(need_init):
            self._ref_height[need_init] = current_height[need_init]
            self._steps_since_ref[need_init] = 0
            self._initialized[need_init] = True

        self._steps_since_ref += 1
        window_closed = self._steps_since_ref >= self.window_steps

        height_gain = current_height - self._ref_height
        stalled = window_closed & (height_gain < min_height_gain)

        if torch.any(window_closed):
            self._ref_height[window_closed] = current_height[window_closed]
            self._steps_since_ref[window_closed] = 0

        return stalled


def orientation_upright_exp(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
    """Graded positive reward for upright orientation, mirroring `height_tracking_exp`'s exact exp-kernel
    shape -- complements `flat_orientation_l2` (a pure penalty with no corresponding positive pull) the
    same way `height_tracking_exp` complements `base_height_l2` in the other asimov1 tasks.

    User's own insight after 5 straight runs showed height/orientation both improving but never converting
    into a held stand: height already gets a strong, graded pull toward the goal via its own exp-kernel
    bonus; orientation only had the penalty's much flatter gradient once already quite tilted (an L2
    penalty barely distinguishes "very tilted" from "extremely tilted" -- both just keep growing, no
    sharpened pull specifically in the "getting close to upright" zone that matters most for actually
    finishing the climb). Added ALONGSIDE `flat_orientation_l2`, not replacing it -- keep the penalty's
    unbounded punishment for getting worse, add this reward's sharp gradient for getting close to upright.

    Uses the exact same underlying quantity as `flat_orientation_l2` (squared XY magnitude of the
    projected gravity vector), just reshaped through an exp kernel instead of returned as a raw penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tilt_sq = torch.sum(torch.square(asset.data.projected_gravity_b.torch[:, :2]), dim=1)
    return torch.exp(-tilt_sq / std**2)


def recovery_pose_alignment_exp(
    env: ManagerBasedRLEnv,
    target_joint_pos: list[float],
    std: float,
    gate_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Graded reward for ending the recovery in a joint configuration close to `target_joint_pos` (the
    WALKING policy's own default/init pose, see `WALKING_HANDOFF_TARGET_JOINT_POS`'s own comment) -- NOT
    just "any valid standing pose" the way `height_tracking`/`stability_hold`/`orientation_upright` alone
    require. Added per explicit user insight after watching real phase-stitch handoff failures: the good
    recoveries looked great, but failures consistently traced back to the recovery's own final pose, not
    anything visible at the root-body level (height/upright/ang_speed all looked statistically identical
    between successful and failed handoffs in direct gate-value comparisons) -- this term targets exactly
    that invisible joint-space gap.

    GATED by height (and smoothly scaled by `_upright_factor`) so this only matters once the robot is
    already close to standing -- same reasoning as `stability_hold`'s own gating, applying this
    unconditionally would actively fight the early get-up motion, which necessarily passes through joint
    configurations nowhere near a walking-ready stance. `gate_height` mirrors the phase-stitch script's own
    `STAND_MIN_PELVIS_HEIGHT = 0.9 * STANDING_HEIGHT` threshold so the reward and the real inference-time
    handoff gate roughly agree on what "close to standing" means.

    `std` is an unmeasured placeholder (no prior data on what a "close enough" aggregate joint-space L2
    distance looks like for this specific pose target) -- flagged for tuning off the first real run's
    logged magnitude, same discipline as every other placeholder in this file.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target = torch.tensor([target_joint_pos], device=asset.data.joint_pos.torch.device)
    pose_error_sq = torch.sum(torch.square(asset.data.joint_pos.torch - target), dim=1)
    alignment = torch.exp(-pose_error_sq / std**2)
    height_gate = (asset.data.root_pos_w.torch[:, 2] > gate_height).float()
    return alignment * height_gate * _upright_factor(asset)


def recovery_hip_symmetry_l2(
    env: ManagerBasedRLEnv,
    gate_height: float,
    hip_roll_weight: float,
    hip_yaw_weight: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalizes LEFT-RIGHT hip asymmetry at the terminal pose specifically -- a sharper, more targeted
    follow-up to `recovery_pose_alignment_exp` above, added after a direct per-joint/per-handoff diagnostic
    found that `recovery_pose_alignment_exp`'s own aggregate L2-across-23-joints formulation was diluting
    exactly this signal: a single badly-off hip_roll joint barely moves a sum that already includes 22
    other joints, so the policy wasn't meaningfully discouraged from occasionally ending with a lopsided
    hip even while everything else looked fine.

    CONFIRMED the real failure mode, not guessed: logged the full joint-position vector at every real
    phase-stitch RECOVER->WALK handoff (success and fail) across a live validation run, computed each
    joint's z-score against the success population, AND the left+right asymmetry (mirror-symmetric poses
    satisfy left == -right per this project's own sign convention, so left+right measures genuine
    imbalance, not just one side's absolute offset). Result: 2 of 6 observed failures showed a hip_roll
    asymmetry z-score of +9.0 and +14.9 against the success population (std ~0.024 rad) -- a massive,
    specific outlier, far beyond anything any other joint or the other 4 failures showed. hip_yaw showed
    the same pattern more weakly (max z ~3.9), included here too at a lower relative weight.

    GATED identically to `recovery_pose_alignment_exp` (height + upright-factor) for the same reason --
    asymmetric hip angles are completely normal and necessary during the early get-up motion, this should
    only matter once already close to standing.

    `asset_cfg.joint_ids` MUST resolve in exactly this order: [left_hip_roll, right_hip_roll, left_hip_yaw,
    right_hip_yaw] -- enforced by the caller passing `joint_names` in that literal order with
    `preserve_order=True` (same convention this project already uses for feet_separation/feet_crossed).
    `hip_roll_weight`/`hip_yaw_weight` are a 3:1 ratio reflecting the observed signal strength (hip_roll's
    own z-scores were roughly 3-4x hip_yaw's) -- a reasoned starting ratio, not a measured optimum.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    hip_roll_asym_sq = torch.square(joint_pos[:, 0] + joint_pos[:, 1])
    hip_yaw_asym_sq = torch.square(joint_pos[:, 2] + joint_pos[:, 3])
    penalty = hip_roll_weight * hip_roll_asym_sq + hip_yaw_weight * hip_yaw_asym_sq
    height_gate = (asset.data.root_pos_w.torch[:, 2] > gate_height).float()
    return penalty * height_gate * _upright_factor(asset)


def recovery_knee_bend_l2(
    env: ManagerBasedRLEnv,
    gate_height: float,
    target_left: float,
    target_right: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalizes knee-bend deviation from the WALKING policy's own exact target values -- the SAME
    dilution problem as `recovery_hip_symmetry_l2` above, now confirmed for the knees specifically (not
    guessed): `recovery_pose_alignment_exp` logs near its max weighted value during training, but a
    direct re-check of the logged number (the training log reports the WEIGHTED average, i.e.
    raw_term * weight -- dividing back out the weight=2.0 gives a much more modest ~0.48 raw average) and
    a live per-handoff diagnostic BOTH independently confirm the knees sit near 0.0 in both successful and
    failed handoffs, nowhere near `WALKING_HANDOFF_TARGET_JOINT_POS`'s bent-knee target (0.38 left / -0.38
    right) -- directly matching the user's own live observation ("recovery ends with a more knees straight
    pose... their knees are super straight"). The aggregate 23-joint L2 sum in
    `recovery_pose_alignment_exp` was diluting this specific pair's error the same way it diluted hip_roll
    before `recovery_hip_symmetry_l2` was added -- same root cause, same fix pattern, now applied to a
    different joint pair found via a different signal (visual inspection + the weight-division math, not
    the asymmetry-diagnostic script this time).

    Unlike `recovery_hip_symmetry_l2` (penalizes a SYMMETRY violation, left+right != 0), this penalizes
    deviation from an ABSOLUTE target each side (straight-knee is a real, consistent, single failure mode
    here, not two sides disagreeing with each other) -- hence separate `target_left`/`target_right` params
    instead of a symmetry formula.

    GATED identically to `recovery_pose_alignment_exp`/`recovery_hip_symmetry_l2` (height + upright-factor)
    for the same reason -- straight knees are normal and necessary during the early climb/push-off phase,
    this should only matter once already close to standing.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    error_sq = torch.square(joint_pos[:, 0] - target_left) + torch.square(joint_pos[:, 1] - target_right)
    height_gate = (asset.data.root_pos_w.torch[:, 2] > gate_height).float()
    return error_sq * height_gate * _upright_factor(asset)


class HeadGroundTermination(ManagerTermBase):
    """Terminates if the tracked upper-body bodies (head/shoulder-region proxy -- same body group as
    `upper_body_ground_penalty`) are still below `min_height` AFTER an initial grace period.

    Direct response to the A-pose persisting across 5 straight runs despite the graded
    `upper_body_ground_penalty` alone -- user's own explicit request for real teeth: "almost want to
    terminate if the head touches the ground."

    The grace period is what makes a hard termination here safe where `StalledRecoveryTermination` above
    was NOT safe to add from iteration 0: that one judged "has enough progress happened," which is
    universally true/false for a policy with no learned behavior yet (so it fired on 100% of episodes
    regardless of parameters). This one judges a simple, momentary geometric fact (are these bodies up or
    down) that's naturally FALSE for most of a random policy's early steps too -- but gated behind a grace
    period, so the one thing every single episode is guaranteed to do (settle somewhere after the initial
    random-orientation drop) never counts against it. Only genuinely being back down there AFTER having
    had real time to not be does.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        grace_period = cfg.params["grace_period"]
        self.grace_steps = max(1, round(grace_period / env.step_dt))
        self._episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._episode_steps[env_ids] = 0

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, grace_period: float, min_height: float
    ) -> torch.Tensor:
        self._episode_steps += 1
        tracked_height = self.asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2].mean(dim=1)
        past_grace = self._episode_steps > self.grace_steps
        return past_grace & (tracked_height < min_height)


def final_pose_phase_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, standing_height: float, transition_width: float = 0.15
) -> torch.Tensor:
    """Penalizes deviation from the robot's own default joint pose (its intended standing configuration --
    asimov-1's all-zero joint pose IS its standing pose, confirmed from the asset's own init_state) -- but
    ONLY once height is already near standing_height, via the same height-gated phase mechanism as
    `knee_bend_phase_bonus`.

    User's own insight: `height_tracking`/`orientation_upright`/`stability_hold` all check WHERE the pelvis
    is and HOW TILTED the torso is, but none of them check whether the actual JOINT CONFIGURATION resembles
    a correct, stable standing pose -- the robot could satisfy height+tilt with legs crossed, asymmetric
    knees, or arms still flailing, and none of those three terms would know the difference.

    CANNOT be active throughout the whole episode (unlike a stand/walk task's own constant
    `joint_deviation_l1` terms) -- the robot NEEDS to deviate substantially from this exact pose during the
    early recovery phase (bent knees, raised arms for leverage), so an unconditional pull toward
    `default_joint_pos` from frame 1 would fight the very recovery motion this task is trying to discover.
    Phase-gated the same way as `knee_bend_phase_bonus` specifically to avoid a "do nothing, stay near
    default pose" exploit: deviation is literally ~0 at the very start of every episode (before the policy
    has done anything), so an ungated version of this would pay the robot for simply not moving.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = torch.sum(
        torch.abs(asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]),
        dim=1,
    )
    height = asset.data.root_pos_w.torch[:, 2]
    phase = torch.clamp((height - (standing_height - transition_width)) / transition_width, min=0.0, max=1.0)
    return deviation * phase


def sit_torso_alignment_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sit_height: float, band_width: float = 0.15
) -> torch.Tensor:
    """Penalizes the non-leg joints (waist_yaw, hip_yaw, hip_roll, shoulders, elbows, wrists, ankle_roll)
    deviating from their default (facing-forward, symmetric) pose -- but ONLY while near the SIT_HEIGHT
    waypoint, via a Gaussian BAND gate centered on `sit_height` (not the one-sided ramp
    `final_pose_phase_penalty`/`knee_bend_phase_bonus` use toward STANDING_HEIGHT -- sitting is a transient
    waypoint the robot passes THROUGH on the way to standing, not a final target to settle at and stay
    above, so this gate must turn back OFF once the robot climbs on past it, unlike the one-sided versions).

    Direct answer to the user's own live observation after run 7: once robots reliably reach the sitting
    waypoint, the torso is twisted/rotated (improved over that run from fully-backwards to half-cocked-
    sideways, never fully resolved). `final_pose_phase_penalty` doesn't help here -- it's gated on proximity
    to STANDING_HEIGHT (phase ~0 this low) and is effectively off during sitting.

    Deliberately targets ONLY the joints that should sit at their default value regardless of sit-vs-stand
    (waist_yaw/hip_yaw/hip_roll/shoulders/elbows/wrists/ankle_roll -- a facing-forward, symmetric torso is
    correct whether sitting or standing) -- EXCLUDES hip_pitch/knee/ankle_pitch, the joints that MUST
    deviate from default to actually sit. This means the reward can reuse `default_joint_pos` directly as
    the target (same array `final_pose_phase_penalty` uses), no new/invented seated-pose joint array needed
    -- just a different joint SUBSET and a different height GATE shape.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = torch.sum(
        torch.abs(asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]),
        dim=1,
    )
    height = asset.data.root_pos_w.torch[:, 2]
    gate = torch.exp(-torch.square((height - sit_height) / band_width))
    return deviation * gate


# attribute on the env holding the current assist-force scale in [0, 1] -- written by AssistCurriculum,
# read every step by apply_assist_force.
ASSIST_SCALE_ATTR = "asimov1_assist_scale"

# attributes holding apply_assist_force's own progress-window state (reference height snapshot + the
# common_step_counter it was taken at) -- see that function's own docstring for the progress-gate mechanism.
ASSIST_PROGRESS_REF_ATTR = "asimov1_assist_progress_ref_height"
ASSIST_PROGRESS_WINDOW_START_ATTR = "asimov1_assist_progress_window_start"


def apply_assist_force(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    asset_cfg: SceneEntityCfg,
    max_weight_fraction: float,
    min_upright: float = 0.3,
    progress_gate_min_height: float = 0.1,
    progress_window_steps: int = 180,
    min_progress: float = 0.03,
) -> None:
    """Upward lift force on `asset_cfg`'s body = (current assist scale) * max_weight_fraction * robot weight
    * an UPRIGHT GATE * a PROGRESS GATE.

    Upright gate: zero while badly tilted, ramping (squared) up to full lift once the robot is already
    within `min_upright` of vertical.

    Ported from OpenHorizon's HoST-inspired assist-force curriculum (openhorizon-labs/asimov-v1-policies,
    `src/asimov_policies/tasks/getup/mdp.py` `apply_assist_force`/`assist_curriculum`) -- their getup policy
    reaches 100% sim success using exactly this mechanism, not reward-shaping alone: physically assist
    standing up early in training, then decay the assist as the measured success rate rises. Worth porting
    because 7 straight runs here steadily improved height/orientation/sitting-waypoint metrics but NEVER
    broke `stability_hold` away from ~0 -- strong evidence the remaining gap needs a different KIND of
    intervention than more reward shaping, and this is a real, verified-working example of exactly that.

    **The upright gate was added after run 8's FIRST launch (unconditional lift, matching OpenHorizon's own
    literal implementation) made the A-pose problem WORSE, not better -- confirmed live.** At iteration
    697/1500, `assist/scale` was still pinned at 1.0 (zero decay the whole run) while `height_tracking`
    (1.52) and `sit_height_tracking` (1.99, maxed) were both artificially high despite `torso_upright`
    sitting at -0.61 (worse than run 7's final -0.45) and `assist/success_rate` staying exactly 0.0 across
    ~31 full episodes. Root cause: `height_tracking`/`sit_height_tracking` only check pelvis Z, with no
    orientation precondition -- an unconditional external lift on the pelvis pays out on both of those
    reward terms regardless of body orientation, so a robot resting in an A-pose (head/shoulders down,
    pelvis as the tripod's high point) gets its pelvis height propped up for free, directly subsidizing the
    exact resting exploit `upper_body_ground_penalty` already fights, rather than helping escape it. OpenHorizon
    doesn't have this failure mode visibly because their `height_progress` reward also tracks HEAD height
    (which stays low in an A-pose-style posture) -- we have no equivalent reward (no head rigid body on this
    asset, only a penalty-side proxy, `upper_body_ground_penalty`), so gating the force itself at the source
    is the more direct fix here. `min_upright=0.3` (`-gravity_z` -- 1.0 fully vertical, 0.0 horizontal/lying
    on its side) means roughly anything tilted more than ~70 degrees from vertical (including fully prone)
    gets ZERO lift; above that it ramps up quadratically (sharper cutoff than linear) to full assist at fully
    upright -- the intent is "help finish the last stretch of standing once you're already oriented roughly
    correctly," not "prop up pelvis height in any posture."

    Progress gate: added after run 9 plateaued for the back half of the run (iteration 817->1499: height_
    tracking 0.55->0.52, sit_height_tracking 1.84->1.83, assist/scale pinned at 1.0 the ENTIRE run) -- the
    upright gate correctly stopped the A-pose exploit, but let a NEW comfortable rest state through: sitting
    bolt-upright satisfies the upright gate fully, so the robot gets the full lift forever just for sitting
    still, with nothing pushing it to spend extra effort climbing further. Fixed the same way `limb_pushoff_
    bonus`/`knee_bend_phase_bonus` were already fixed earlier this session -- gate on ACTIVE EFFORT
    (continuing to climb), not a static condition alone. Tracks height against a periodically-refreshed
    reference (global cadence, `progress_window_steps` control steps -- fine since this task's only
    termination is `time_out`, so every env resets in sync, same simplifying assumption `AssistCurriculum`
    already uses); if height hasn't risen by at least `min_progress` since that reference, the gate fades.

    Deliberately does NOT enforce this below `progress_gate_min_height` -- full assist regardless of recent
    progress while still near the floor, so the early climb (the part assist is MOST needed for) is never
    handicapped by a progress check. This is the same class of hazard that killed `StalledRecoveryTermination`
    and `HeadGroundTermination` earlier this session (a progress/fact check applied before any skill exists
    to pass it with fires universally) -- but the consequence here is far milder (less bonus assist, not an
    ended episode), and restricting it to ABOVE a height the robot has already demonstrably reached removes
    most of that risk: by the time height exceeds `progress_gate_min_height`, the policy has already shown
    it can produce that much climb on its own at least once.

    Applied every control step via an `interval` EventTerm with `interval_range_s=(env_step_dt, env_step_dt)`
    and `is_global_time=True` -- Isaac Lab's EventManager has no literal "every step" mode the way mjlab
    (OpenHorizon's framework) does, but setting the interval bounds equal to the control step makes the
    countdown hit its resample threshold every single step (confirmed from `event_manager.py`'s own interval
    logic: `time_left -= dt` then resample+fire the instant `time_left < 1e-6`) -- functionally identical.

    Uses `asset.permanent_wrench_composer.set_forces_and_torques_index` (the current, non-deprecated
    external-wrench API, confirmed via the quadcopter task's own real usage of it) rather than the
    deprecated `set_external_force_and_torque` wrapper.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    scale = float(getattr(env, ASSIST_SCALE_ATTR, 0.0))
    height = asset.data.root_pos_w.torch[:, 2]

    upright = torch.clamp(-asset.data.projected_gravity_b.torch[:, 2], min=0.0, max=1.0)
    upright_gate = torch.clamp((upright - min_upright) / (1.0 - min_upright), min=0.0, max=1.0) ** 2

    # lazily init + periodically refresh the progress-window reference (global cadence -- see docstring).
    if not hasattr(env, ASSIST_PROGRESS_REF_ATTR) or (
        env.common_step_counter - getattr(env, ASSIST_PROGRESS_WINDOW_START_ATTR) >= progress_window_steps
    ):
        setattr(env, ASSIST_PROGRESS_REF_ATTR, height.clone())
        setattr(env, ASSIST_PROGRESS_WINDOW_START_ATTR, env.common_step_counter)
    ref_height = getattr(env, ASSIST_PROGRESS_REF_ATTR)
    progress = torch.clamp(height - ref_height, min=0.0)
    progress_factor = torch.clamp(progress / min_progress, min=0.0, max=1.0)
    # blend: full pass (1.0) below progress_gate_min_height, ramping to progress_factor above it -- so the
    # progress requirement only ever engages once the robot has already climbed past the floor.
    height_gate = torch.clamp((height - progress_gate_min_height) / 0.05, min=0.0, max=1.0)
    progress_gate = torch.lerp(torch.ones_like(progress_factor), progress_factor, height_gate)

    total_mass = asset.data.body_mass.torch.sum(dim=1)  # (num_envs,)
    forces = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    forces[:, 0, 2] = scale * max_weight_fraction * total_mass * 9.81 * upright_gate * progress_gate
    torques = torch.zeros_like(forces)
    asset.permanent_wrench_composer.set_forces_and_torques_index(forces=forces, torques=torques, body_ids=asset_cfg.body_ids)


class AssistCurriculum(ManagerTermBase):
    """Decays the assist-force scale (see `apply_assist_force`) as the measured standing-success rate rises.

    Direct port of OpenHorizon's `assist_curriculum` class. Counts episode-ends over a window of
    `window_steps` env steps (our task's only termination is `time_out`, so every env resets in sync every
    `episode_length_s` -- unlike OpenHorizon's possibly-staggered resets, so `window_steps` just needs to be
    at most one episode's worth of steps to catch each synchronized reset burst). At the end of each window,
    if the fraction of ended episodes that were "standing" (pelvis above `min_pelvis_height`, torso upright
    past `min_upright`, both feet in contact) exceeds `success_threshold`, the scale drops by `decay_step`
    (floor 0). "Standing" here is a one-step check at the moment of reset, not a held-duration success
    metric like `stability_hold`'s own counter -- deliberately simpler, matching OpenHorizon's own
    definition (`cmdp.standing_mask`), not reusing `stability_hold`'s internal state (keeps this curriculum
    decoupled from a specific reward term's own bookkeeping).

    The scale is NOT stored in checkpoints (matches OpenHorizon's own design choice) -- a resumed run starts
    again from `initial_scale` and decays quickly if the policy is already good enough to clear the
    threshold immediately.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._ended = 0
        self._stood = 0
        self._window_start = 0
        self._last_rate = 0.0
        setattr(env, ASSIST_SCALE_ATTR, float(cfg.params["initial_scale"]))

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | slice,
        initial_scale: float,
        decay_step: float,
        success_threshold: float,
        window_steps: int,
        min_pelvis_height: float,
        min_upright: float,
        feet_sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> dict[str, float]:
        del initial_scale
        # skip the very first reset (before any step has happened) -- nobody has had a chance yet.
        if env.common_step_counter > 0:
            asset: Articulation = env.scene[asset_cfg.name]
            pelvis_height = asset.data.root_pos_w.torch[env_ids, 2]
            upright = -asset.data.projected_gravity_b.torch[env_ids, 2]
            sensor = env.scene.sensors[feet_sensor_cfg.name]
            forces = sensor.data.net_forces_w.torch[env_ids][:, feet_sensor_cfg.body_ids, :]
            feet_in_contact = (torch.norm(forces, dim=-1) > 1.0).all(dim=-1)
            standing = (pelvis_height > min_pelvis_height) & (upright > min_upright) & feet_in_contact

            self._ended += int(standing.numel())
            self._stood += int(standing.sum().item())
            if env.common_step_counter - self._window_start >= window_steps and self._ended > 0:
                self._last_rate = self._stood / self._ended
                scale = float(getattr(env, ASSIST_SCALE_ATTR))
                if self._last_rate > success_threshold and scale > 0.0:
                    setattr(env, ASSIST_SCALE_ATTR, max(0.0, scale - decay_step))
                self._ended = self._stood = 0
                self._window_start = env.common_step_counter

        return {"scale": float(getattr(env, ASSIST_SCALE_ATTR)), "success_rate": self._last_rate}


class TurnBiasedVelocityCommand(UniformVelocityCommand):
    """Same uniform SE(2) velocity command as the stock `UniformVelocityCommand`, except each resample
    is drawn from one of two episode "modes" instead of sampling all three axes independently every
    time: TURN mode (probability `cfg.turn_episode_prob`) samples ang_vel_z from its full configured
    range while narrowing lin_vel_x/lin_vel_y to a small near-zero band; LINEAR mode (the remaining
    probability) does the reverse -- full lin_vel_x/y range, narrow ang_vel_z.

    Added 2026-10-20 directly off live evidence from the asimov1 go-to-point v0 executor
    (asimov1_phase_stitch/goto.py): "near-zero forward speed, high turn rate, sustained for several
    seconds" is a real, under-trained corner of this policy's command space. Under the stock command's
    independent-per-axis uniform sampling, that specific combination only ever occurs by chance (all 3
    axes landing in a narrow joint region simultaneously) and never for long, since every resample
    redraws all three axes fresh. This class oversamples exactly that combination (and its mirror,
    committed straight walking) instead of relying on chance to cover it -- a direct curriculum
    response to a diagnosed gap, not a guess.
    """

    cfg: TurnBiasedVelocityCommandCfg

    def _resample_command(self, env_ids):
        n = len(env_ids)
        is_turn_episode = torch.empty(n, device=self.device).uniform_(0.0, 1.0) <= self.cfg.turn_episode_prob

        lin_x_full = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.lin_vel_x)
        lin_y_full = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.lin_vel_y)
        ang_z_full = torch.empty(n, device=self.device).uniform_(*self.cfg.ranges.ang_vel_z)
        lin_x_narrow = torch.empty(n, device=self.device).uniform_(*self.cfg.turn_episode_lin_vel_range)
        lin_y_narrow = torch.empty(n, device=self.device).uniform_(*self.cfg.turn_episode_lin_vel_range)
        ang_z_narrow = torch.empty(n, device=self.device).uniform_(*self.cfg.linear_episode_ang_vel_range)

        self.vel_command_b[env_ids, 0] = torch.where(is_turn_episode, lin_x_narrow, lin_x_full)
        self.vel_command_b[env_ids, 1] = torch.where(is_turn_episode, lin_y_narrow, lin_y_full)
        self.vel_command_b[env_ids, 2] = torch.where(is_turn_episode, ang_z_full, ang_z_narrow)

        if self.cfg.heading_command:
            r = torch.empty(n, device=self.device)
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = (
            torch.empty(n, device=self.device).uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        )
        # NOTE: standing-env zeroing of BOTH axes still happens afterward, every step, in the base
        # class's own _update_command -- unchanged and untouched here, so rel_standing_envs continues
        # to mean exactly what it already does regardless of which mode an episode landed in.


@configclass
class TurnBiasedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Config for TurnBiasedVelocityCommand. `turn_episode_lin_vel_range`/`linear_episode_ang_vel_range`
    default to a tight band around zero rather than exactly 0.0 -- a literal constant across an entire
    resample window felt like an unnecessary extra restriction with no clear benefit; a small nonzero
    band keeps some natural diversity within each mode without diluting the intended bias."""

    class_type: type = TurnBiasedVelocityCommand
    turn_episode_prob: float = 0.75
    turn_episode_lin_vel_range: tuple[float, float] = (-0.1, 0.1)
    linear_episode_ang_vel_range: tuple[float, float] = (-0.15, 0.15)
