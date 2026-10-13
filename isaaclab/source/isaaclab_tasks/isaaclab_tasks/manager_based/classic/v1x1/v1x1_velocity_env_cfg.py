# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Omni-directional (velocity-command) locomotion environment for v1x1.

Branches off the single-vector "walk toward a fixed point" task (v1x1_env_cfg.py) into Isaac Lab's other
task style: the policy is given a *commanded* body-frame velocity (lin_vel_x, lin_vel_y, ang_vel_z) each
episode and learns to track whatever it's told, rather than always beelining for one point -- this is the
architecture Anymal/Spot/G1 use, and the prerequisite for eventual arrow-key teleop (a runtime layer that
overwrites the commanded velocity at play time, not a separate training track).

AXIS-MAPPING GOTCHA: track_lin_vel_xy_exp/track_ang_vel_z_exp (isaaclab.envs.mdp) just compare the
commanded velocity against the robot's own raw body-frame velocity (root_lin_vel_b) -- no hardcoded
"forward axis" assumption baked into the math (unlike the old broken FORWARD_VEC_B that caused the
single-vector task's veer bug). But v1x1's real anatomical forward is LOCAL +Y (front legs mount on +Y,
rear on -Y, per the source xacro), not the +X most Isaac Lab robots use -- so the command's lin_vel_x slot
practically means "v1x1's lateral speed" and lin_vel_y means "v1x1's real forward/backward speed." No code
fix needed, just remember this when wiring arrow keys later: up arrow -> lin_vel_y, not lin_vel_x.

Co-located here (not moved to locomotion/velocity/config/v1x1/) so both tasks can share mdp.py's custom
terms (TrotGaitReward, left_right_symmetry_l2, front_rear_symmetry_l2, heading_velocity_l2,
tibia_vertical_l2, foot_clearance_reward) without an import split. Keeps effort/torque control
(JointEffortActionCfg, scale=50.0) -- all of the single-vector task's gait tuning (trot_gait std,
joint-centering weights, symmetry rewards, limits) was developed and validated under effort control, not
the stock velocity family's JointPositionActionCfg.

front_foot_forward is intentionally DROPPED here (present in v1x1_env_cfg.py, absent below): it rewards
front feet sitting forward of base_link in the body's own frame, which was correct for forward-only
walking but would actively fight backward/sideways commands. Everything else non-directional carries over
unchanged at its current tuned weight.
"""

from isaaclab_newton.physics import KaminoSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as humanoid_mdp
from isaaclab_tasks.utils import PresetCfg

from . import mdp as v1x1_mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.v1x1 import V1X1_CFG  # isort: skip


@configclass
class V1X1PhysicsCfg(PresetCfg):
    default: PhysxCfg = PhysxCfg(bounce_threshold_velocity=0.2)
    physx: PhysxCfg = PhysxCfg(bounce_threshold_velocity=0.2)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=38,
            nconmax=15,
            cone="pyramidal",
            integrator="implicitfast",
            impratio=1,
        ),
        num_substeps=1,
        debug_mode=False,
    )
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoSolverCfg(
            integrator="moreau",
            use_collision_detector=False,
            sparse_jacobian=True,
            constraints_alpha=0.1,
            padmm_max_iterations=100,
            padmm_primal_tolerance=1e-4,
            padmm_dual_tolerance=1e-4,
            padmm_compl_tolerance=1e-4,
            padmm_rho_0=0.05,
            padmm_eta=1e-5,
            padmm_use_acceleration=True,
            padmm_warmstart_mode="containers",
            padmm_contact_warmstart_method="geom_pair_net_force",
            padmm_use_graph_conditionals=False,
            collision_detector_pipeline="unified",
            collision_detector_max_contacts_per_pair=8,
        ),
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=True,
    )


@configclass
class V1X1ContactSensorCfg(PresetCfg):
    default: PhysXContactSensorCfg = PhysXContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    physx: PhysXContactSensorCfg = PhysXContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    newton_mjwarp: NewtonContactSensorCfg = NewtonContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a v1x1 robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot
    robot = V1X1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = V1X1ContactSensorCfg()

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # Ranges are a starting estimate -- v1x1's proven committed-gait speed (single-vector task) is
    # ~0.5-0.6 m/s. heading_command=False for a simpler direct-sampling first pass rather than the
    # heading-P-controller variant. Per the axis-mapping gotcha above, lin_vel_x here is v1x1's LATERAL
    # speed and lin_vel_y is v1x1's real forward/backward speed.
    # lin_vel_x/y range RAISED 0.5 -> 0.75 m/s (50%) per feedback wanting a speed increase now that the gait
    # is otherwise solid -- ang_vel_z range left unchanged, this is specifically about linear speed.
    # resampling_time_range LOWERED 10.0 -> 5.0 alongside the heading_hold_bonus weight bump (see mdp.py's
    # HeadingHoldBonus): that reward's reference direction is recaptured every time the command resamples,
    # so shortening the window directly bounds how long ANY drift can accumulate before getting reset with a
    # fresh, correct reference -- a genuinely complementary change to the same mechanism, not an unrelated
    # second variable (unlike when this same idea was floated before heading_hold_bonus existed).
    # NOTE: a color-coded (green=straight/orange=turning) debug-vis arrow was attempted here via a custom
    # V1X1UniformVelocityCommandCfg, but its module-level VisualizationMarkersCfg construction in mdp.py
    # caused a persistent USD/Kit startup crash (libusd_tf.so TfRegistryManager) that survived a reboot and
    # cache-clearing -- removed 2026-09-30, see mdp.py's note at the old location for full diagnosis. Back to
    # the stock command class; revisit the color-coded arrow later with a lazier construction if still wanted.
    base_velocity = base_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        heading_command=False,
        rel_standing_envs=0.02,
        debug_vis=True,
        ranges=base_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.75, 0.75), lin_vel_y=(-0.75, 0.75), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = base_mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=50.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        velocity_commands = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, scale=0.2)
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class V1X1VelocityObservationsCfg(PresetCfg):
    default: ObservationsCfg = ObservationsCfg()
    physx: ObservationsCfg = ObservationsCfg()
    newton_mjwarp: ObservationsCfg = ObservationsCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1)/(2) Primary velocity-tracking rewards -- the direction-driving terms for this task, replacing
    # progress/move_to_target from the single-vector task. std=sqrt(0.25) mirrors Isaac Lab's own stock
    # default as a starting point. Weights BUMPED (2.0->3.0, 1.0->1.5) after runs 2 and 3 both showed reward
    # climbing to healthy-looking levels while the robot kept visibly wiggling in place rather than actually
    # translating -- raising these relative to the "free" static rewards (alive, joint-centering terms, which
    # pay out just for existing without any movement risk) increases the real cost of standing still.
    track_lin_vel_xy_exp = RewTerm(
        func=base_mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    # std history: 0.5 (stock default) -> 0.3 (run 7, first attempt at fixing circular drift -- measurably
    # helped, error_vel_yaw improved, but next-morning playback showed a spread of behavior across the
    # population: some individuals track straight, others loose-circle, others hard-turn, suggesting the fix
    # was heading the right direction but hadn't fully converged in just one run). TIGHTENED further 0.3 ->
    # 0.2 (the more aggressive option held back in run 7 for lack of iteration time) to push harder on the
    # same root cause: episodes routinely surviving the full 20s (100% success rate) gives any small constant
    # yaw-rate error enough TIME to compound into a visible circle even though each individual step's penalty
    # looks tiny.
    track_ang_vel_z_exp = RewTerm(
        func=base_mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": 0.2}
    )
    # (2b) Direct velocity-DIRECTION alignment (cosine similarity of actual vs. commanded velocity vectors,
    # NOT body orientation -- see v1x1_mdp.velocity_direction_alignment docstring for why this is safe for
    # strafing/backward commands where heading_to_velocity was not). Added for the same reason as the weight
    # bumps above: isolates the "pick a direction and commit to it" signal from the combined magnitude+
    # direction L2 error that track_lin_vel_xy_exp already provides, which apparently wasn't decisive enough
    # on its own across runs 2 and 3.
    velocity_direction_alignment = RewTerm(
        func=v1x1_mdp.velocity_direction_alignment, weight=1.0, params={"command_name": "base_velocity"}
    )
    # (2c) Direct fix for the "slowly drifting/spinning in circles" finding from runs 6-8. Unlike
    # velocity_direction_alignment above (which reinterprets the command relative to the robot's CURRENT,
    # possibly-drifted body frame every step, so it has no restoring force), this term captures a REFERENCE
    # direction in WORLD frame at the moment each command is issued and rewards holding close to THAT fixed
    # reference over time -- see v1x1_mdp.HeadingHoldBonus docstring for the full "car with slightly turned
    # wheels" reasoning. First run (weight=1.5) showed the mechanism genuinely engaging (raw value
    # consistently positive, 0.03-0.65) but not decisively winning for most of the population -- playback
    # still showed most robots circling. RAISED 1.5 -> 3.0 (matching track_lin_vel_xy_exp's weight, putting
    # world-direction-holding on equal footing with the primary tracking reward rather than a secondary
    # nudge) per feedback wanting the circling actually gone, not just reduced.
    # Weight history: 1.5 -> 3.0 -> 4.0 -> 5.0 (this round) -- run 12 ("most still drift") showed the soft
    # reward alone wasn't decisive for the bulk of the population even with the hard termination tightened to
    # 15 deg, so pushing this further alongside that tightening rather than relying on the termination alone.
    heading_hold_bonus = RewTerm(
        func=v1x1_mdp.HeadingHoldBonus,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "min_command_speed": 0.1,
            # gated off for commands that also ask for real rotation -- see HeadingHoldBonus docstring,
            # added after run 11's postmortem found this reward (and the termination below) was punishing
            # legitimately curving travel during genuine turn commands as if it were erroneous drift.
            "max_ang_vel_for_hold": 0.2,
        },
    )
    # (3)/(4) Companion penalties discouraging vertical bounce / excess roll-pitch rate while tracking.
    lin_vel_z_l2 = RewTerm(func=base_mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-0.05)
    # (5) Stay alive bonus
    alive = RewTerm(func=humanoid_mdp.is_alive, weight=0.3)
    # (6) Penalty for large action commands
    action_l2 = RewTerm(func=humanoid_mdp.action_l2, weight=-0.005)
    # (7) Penalty for energy consumption
    energy = RewTerm(func=humanoid_mdp.power_consumption, weight=-0.05, params={"gear_ratio": {".*": 100.0}})
    # (8) Penalty for reaching close to joint limits
    joint_pos_limits = RewTerm(
        func=humanoid_mdp.joint_pos_limits_penalty_ratio,
        weight=-0.1,
        params={"threshold": 0.99, "gear_ratio": {".*": 100.0}},
    )
    # (9) Reward for actually taking steps. Threshold history: 0.75 (single-vector task) -> 0.2 (run 4, unstuck
    # the wiggling local optimum) -> 0.35 (run 5) -> 0.7 (run 6) -> 1.0 (run 7). Each raise recovered faster
    # than the last (run 7's reward barely dipped and bounced right back), so there's still real headroom.
    # RAISED again 1.0 -> 1.3 per feedback wanting step height/stride length "even more."
    # Threshold history: 0.75 -> 0.2 (run 4) -> 0.35 (run 5) -> 0.7 (run 6) -> 1.0 (run 7) -> 1.3 (run 8) ->
    # 1.6 (run 12). RAISED again 1.6 -> 2.0 per feedback wanting a fourth stride-length increase.
    feet_air_time = RewTerm(
        func=v1x1_mdp.feet_air_time_unconditional,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg1_link4", "leg2_link4", "leg3_link4", "leg4_link4"]
            ),
            "threshold": 2.0,
        },
    )
    # (10) Penalty for the chassis touching the ground (belly-flopping).
    base_contact = RewTerm(
        func=base_mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]), "threshold": 1.0},
    )
    # (11) Trot-gait shaping. leg1=rear-left, leg2=rear-right, leg3=front-left, leg4=front-right. Gated on
    # the robot's own measured body velocity (not a command), so it naturally stays quiet during near-zero-
    # velocity commands too. velocity_threshold LOWERED from the single-vector task's 0.5 m/s -> 0.15 m/s --
    # run 2's postmortem found trot_gait sitting at EXACTLY 0.0000 the entire run: 0.5 m/s worked for the
    # single-vector task because progress_reward reliably drove it to real walking speed early, giving
    # trot_gait something to reward once moving; this task's policy never reached 0.5 m/s (stuck wiggling in
    # place), so trot_gait never activated even once, giving it zero chance to shape early stepping attempts.
    # 0.15 m/s (30% of the max commanded speed of 0.5 m/s) should let it engage on genuine-but-modest movement
    # instead of requiring speed the policy hasn't learned yet.
    trot_gait = RewTerm(
        func=v1x1_mdp.TrotGaitReward,
        weight=1.5,
        params={
            "std": 0.06,
            "max_err": 0.2,
            "velocity_threshold": 0.15,
            "synced_feet_pair_names": (("leg3_link4", "leg2_link4"), ("leg4_link4", "leg1_link4")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    # (12)-(14) Keep the coxa/femur-end/tibia joints close to their defaults, carried over unchanged.
    coxa_orthogonal = RewTerm(
        func=humanoid_mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg.*_joint1"])},
    )
    femur_end_centered = RewTerm(
        func=humanoid_mdp.joint_deviation_l1,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg.*_joint3"])},
    )
    tibia_vertical = RewTerm(
        func=v1x1_mdp.tibia_vertical_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["leg1_link4", "leg2_link4", "leg3_link4", "leg4_link4"]
            )
        },
    )
    tibia_extension = RewTerm(
        func=humanoid_mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg.*_joint4"])},
    )
    # (15) Keep the chassis level with the ground, carried over unchanged.
    chassis_level = RewTerm(func=humanoid_mdp.flat_orientation_l2, weight=-1.6)
    # (16) heading_to_velocity DROPPED for this task (was carried over from the single-vector task in run 1/2,
    # weight -1.5). Unlike the single-vector task -- which only ever needed one direction, so "face the way
    # you're moving" was always correct -- v1x1 is symmetric across all 4 legs (left_right_symmetry AND
    # front_rear_symmetry are both enforced), so it should be equally capable of walking forward, backward, or
    # sideways (crab-walking) without turning to face the commanded direction first. heading_to_velocity
    # actively fights that: it would force a turn-then-walk pattern for any non-forward command, adding
    # unnecessary difficulty for a robot whose morphology doesn't require it. Its measured contribution was
    # small in run 2 (~-0.02 raw), so it likely wasn't the dominant cause of that run's stuck-wiggling
    # failure (see feet_air_time/trot_gait threshold fixes above for the bigger structural culprits), but it's
    # still fighting the wrong objective for this task and has no reason to stay.
    # (17) Nudge base_link height, carried over unchanged.
    base_height = RewTerm(func=humanoid_mdp.base_height_l2, weight=-0.3, params={"target_height": 0.15})
    # (18)/(19) Left/right and front/rear joint-symmetry penalties -- the fix for the recurring per-leg
    # asymmetry bug, still fully general for omni-directional commands. Weights TIGHTENED further -0.35 ->
    # -0.5 (previously -0.2 -> -0.35 for run 5) -- part of the final round's combined fix for the "slowly
    # drifting in circles" finding: residual leg asymmetry likely contributes a small constant yaw torque
    # bias on top of the tracking reward's own tolerance for small errors (see track_ang_vel_z_exp std
    # tightening above), so squeezing this further should help alongside that fix, not instead of it.
    left_right_symmetry = RewTerm(func=v1x1_mdp.left_right_symmetry_l2, weight=-0.5)
    front_rear_symmetry = RewTerm(func=v1x1_mdp.front_rear_symmetry_l2, weight=-0.5)
    # (20) Foot swing clearance. target_height DOUBLED 0.04 -> 0.08 per feedback wanting step height increased
    # alongside stride length -- std widened proportionally 0.02 -> 0.03 (same ~2x target/std ratio as before)
    # so the reward gradient stays similarly shaped around the new, higher target rather than becoming overly
    # narrow/punishing relative to the bigger target.
    # target_height RAISED again 0.12 -> 0.16, std scaled proportionally 0.045 -> 0.06 (same ratio) per
    # feedback wanting a fourth step-height increase alongside the stride-length bump above.
    foot_clearance = RewTerm(
        func=v1x1_mdp.foot_clearance_reward,
        weight=0.3,
        params={
            "target_height": 0.16,
            "std": 0.06,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["leg1_link4", "leg2_link4", "leg3_link4", "leg4_link4"]
            ),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP -- copied unchanged from v1x1_env_cfg.py (all safety terms, direction-
    agnostic)."""

    time_out = DoneTerm(func=humanoid_mdp.time_out, time_out=True)
    torso_height = DoneTerm(func=humanoid_mdp.root_height_below_minimum, params={"minimum_height": 0.04})
    orientation = DoneTerm(func=humanoid_mdp.bad_orientation, params={"limit_angle": 2.0})
    foot_air_time = DoneTerm(
        func=v1x1_mdp.foot_air_time_exceeded,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg1_link4", "leg2_link4", "leg3_link4", "leg4_link4"]
            ),
            "max_air_time": 1.5,
        },
    )
    foot_contact_time = DoneTerm(
        func=v1x1_mdp.foot_contact_time_exceeded,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg1_link4", "leg2_link4", "leg3_link4", "leg4_link4"]
            ),
            "max_contact_time": 3.0,
        },
    )
    all_feet_grounded = DoneTerm(
        func=v1x1_mdp.all_feet_grounded_too_long,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg1_link4", "leg2_link4", "leg3_link4", "leg4_link4"]
            ),
            "max_time": 2.0,
        },
    )
    # Bound loosened to +/-0.9 here (vs. +/-0.7 in the single-vector task): that tighter bound is safe for
    # the single-vector lineage because it's the product of a long-converged policy that learned to stay
    # away from the wall, but at fresh iteration-0 this task's default front-coxa stance (leg3/leg4_joint1
    # = +/-0.55) plus reset_robot_joints' up-to-0.2rad reset jitter can reach +/-0.75 -- past +/-0.7 --
    # before the policy has taken a single action. Empirically this pinned coxa_limit at 100% of all
    # terminations from iteration ~14 onward in the first smoke test (episode length collapsing 36 -> 6),
    # a much worse/faster version of the exact "insufficient margin" disaster documented in
    # isaaclab_assets/robots/v1x1.py's init_state comments. +/-0.9 leaves 0.15 rad of real margin against
    # the 0.75 rad worst case (matching that file's own "3x margin" precedent), still well inside the
    # +/-2.356 hardware limit.
    coxa_limit = DoneTerm(
        func=humanoid_mdp.joint_pos_out_of_manual_limit,
        params={"bounds": (-0.9, 0.9), "asset_cfg": SceneEntityCfg("robot", joint_names=["leg.*_joint1"])},
    )
    femur_end_limit = DoneTerm(
        func=humanoid_mdp.joint_pos_out_of_manual_limit,
        params={"bounds": (-1.1, 1.1), "asset_cfg": SceneEntityCfg("robot", joint_names=["leg.*_joint3"])},
    )
    tibia_limit = DoneTerm(
        func=humanoid_mdp.joint_pos_out_of_manual_limit,
        params={"bounds": (-1.4, 1.4), "asset_cfg": SceneEntityCfg("robot", joint_names=["leg.*_joint4"])},
    )
    # NEW this session -- a hard backstop against persistent circular drift, complementing heading_hold_bonus's
    # soft reward (see v1x1_mdp.HeadingDeviationExceeded docstring for the full design rationale). Ends the
    # episode outright (losing ALL remaining reward) if the robot's actual velocity direction stays more than
    # max_angle_deg off its reference direction for a full continuous second -- after a 1s grace period
    # following each resample, so the natural settling transient right after a direction change isn't
    # punished, only genuine persistent circling. v1x1's leg symmetry means it should be able to switch
    # translation direction instantly without rotating base_link at all -- this directly enforces that design
    # intent. max_angle_deg TIGHTENED 30 -> 15 per feedback: with the gate fix in place, this was only
    # catching ~7-8% of terminations (the most extreme circlers), but "most" robots were still visibly
    # drifting somewhat on playback -- moderate drift wasn't severe enough to cross 30 deg, so the hard
    # filter wasn't pressuring the bulk of the population. Halving the threshold should catch moderate
    # drifters too, not just the worst offenders.
    heading_deviation_exceeded = DoneTerm(
        func=v1x1_mdp.HeadingDeviationExceeded,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "max_angle_deg": 15.0,
            # sustained_duration TIGHTENED 1.0 -> 0.5 per explicit feedback ("whatever makes it stricter") --
            # less time allowed continuously off-course before the episode is cut.
            "sustained_duration": 0.5,
            "grace_period": 1.0,
            # gated off for commands that also ask for real rotation -- without this, run 11 pinned at ~100%
            # termination rate and never recovered: a curving travel direction during a genuine turn command
            # is correct execution, not drift, and this was punishing both identically. See mdp.py docstring.
            "max_ang_vel_for_hold": 0.2,
        },
    )


@configclass
class V1X1VelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the v1x1 quadruped omni-directional (velocity-command) locomotion environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=True)
    # Basic settings
    observations: V1X1VelocityObservationsCfg = V1X1VelocityObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = V1X1PhysicsCfg()
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
