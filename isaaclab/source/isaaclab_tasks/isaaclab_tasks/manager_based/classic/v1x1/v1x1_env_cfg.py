# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import KaminoSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

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

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp
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
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=50.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        # target is along world +Y, not +X: v1x1's front legs (leg3/leg4) mount on the body's +Y side and
        # rear legs (leg1/leg2) on -Y, so +Y is the natural front-to-rear quadruped walking axis for this
        # leg layout -- +X would be a sideways/crab-walk direction relative to how the legs are arranged.
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (0.0, 1000.0, 0.0)})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (0.0, 1000.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "joint_wrench",
                    body_names=["leg1_link4", "leg2_link4", "leg3_link4", "leg4_link4"],
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class V1X1ObservationsCfg(PresetCfg):
    default: ObservationsCfg = ObservationsCfg()
    physx: ObservationsCfg = ObservationsCfg()
    newton_mjwarp: ObservationsCfg = ObservationsCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for moving forward -- this is the only term that actually scales with real closing speed
    # (it's literally delta-distance-to-target / dt); weight bumped up hard because it was only contributing
    # ~0.02/step despite already being the top weight, versus up to ~1.0/step from alive+move_to_target
    # combined regardless of how much the robot actually moves -- that imbalance rewarded standing still
    # over walking. move_to_target_bonus in particular is a *heading-alignment* check that saturates at its
    # max value once roughly facing the target, independent of speed, so it does nothing to push for more
    # movement once satisfied -- trimmed down along with alive so bold movement is worth more than safety.
    # Weight bumped 4.0 -> 4.5 -> 5.0 per feedback wanting faster overall movement (this is the term that
    # actually scales with real speed).
    progress = RewTerm(func=mdp.progress_reward, weight=5.0, params={"target_pos": (0.0, 1000.0, 0.0)})
    # (2) Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=0.3)
    # (3) Reward for non-upright posture
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})
    # (4) Reward for base_link heading alignment with the target vector -- weight zeroed out entirely (was
    # 0.6 -> 0.05 -> 0.0) after finding a real structural conflict: this reward (via base_heading_proj)
    # always uses the body's LOCAL +X axis as "forward" (isaaclab_physx's FORWARD_VEC_B is hardcoded to
    # (1,0,0), not configurable, not tied to target_pos). v1x1's legs are mounted along LOCAL Y (front pair
    # +Y, rear pair -Y, per the source xacro), so the leg-propulsion axis is always perpendicular to
    # whatever axis this reward wants facing the target. No choice of target_pos can fix this -- it's a
    # body-frame mismatch, not a world-direction one -- swapping target_pos to (0,1000,0) only moved which
    # world direction got the sideways treatment. Even at a "light nudge" weight of 0.05, a play-mode test
    # of the resulting policy showed a slow, consistent left veer -- a small but structurally-misaligned
    # reward term produces a persistent one-directional yaw bias rather than neutral noise. progress_reward
    # (pure displacement, orientation-agnostic) doesn't have this problem and is kept as the sole direction
    # driver. Term left in place (weight 0.0) rather than deleted, so it's trivial to re-weight if needed.
    move_to_target = RewTerm(
        func=mdp.move_to_target_bonus, weight=0.0, params={"threshold": 0.8, "target_pos": (0.0, 1000.0, 0.0)}
    )
    # (5) Penalty for large action commands
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.005)
    # (6) Penalty for energy consumption
    energy = RewTerm(func=mdp.power_consumption, weight=-0.05, params={"gear_ratio": {".*": 100.0}})
    # (7) Penalty for reaching close to joint limits
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio, weight=-0.1, params={"threshold": 0.99, "gear_ratio": {".*": 100.0}}
    )
    # (8) Reward for actually taking steps -- without this, "stand perfectly still" is a stable local optimum
    # (guaranteed alive/upright reward, zero fall risk) with no pressure to ever attempt a gait. Threshold
    # bumped 0.2 -> 0.32 per feedback wanting bigger strides, not just quick little shuffling steps -- only
    # air-time beyond the threshold pays out, so raising it demands a fuller swing phase per step.
    feet_air_time = RewTerm(
        func=v1x1_mdp.feet_air_time_unconditional,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg1_link4", "leg2_link4", "leg3_link4", "leg4_link4"]
            ),
            # Threshold bumped further 0.32 -> 0.42 per feedback wanting even bigger strides (not huge, just
            # bigger than the current walk) -- same lever as before, demands a fuller swing phase per step.
            # Threshold bumped further 0.42 -> 0.5 -> 0.58 -> 0.75 per feedback wanting air time up "a fair
            # amount" this pass -- bigger jump than the previous small increments.
            "threshold": 0.75,
        },
    )
    # (9) Penalty for the chassis touching the ground (belly-flopping), as a continuous shaping signal
    # complementing the torso_height/orientation terminations, which only fire once things are already bad.
    base_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]), "threshold": 1.0},
    )
    # (10) Trot-gait shaping: reward diagonal foot pairs (front-left+rear-right, front-right+rear-left)
    # stepping in sync with each other and out of sync with the other pair -- gives the policy an explicit
    # hint toward an ordered footfall pattern instead of leaving gait discovery entirely to chance.
    # leg1=rear-left, leg2=rear-right, leg3=front-left, leg4=front-right (per the source URDF/xacro).
    # std/max_err/velocity_threshold mirror Isaac Lab's own Spot trot config as a reasonable starting point.
    # weight bumped 1.0 -> 1.5 and std tightened 0.1 -> 0.06 per feedback wanting the diagonal (X-shaped)
    # footfall pattern enforced more strictly, rather than the looser H-ish pattern a forgiving std still
    # allows -- the underlying math already scores BOTH intra-pair sync AND inter-pair anti-sync (4 cross
    # terms), i.e. it already targets X over H structurally; tightening std just makes it less tolerant of
    # partial/sloppy timing now that the gait is otherwise walking well (was 0.75/~1.0 at last check).
    trot_gait = RewTerm(
        func=v1x1_mdp.TrotGaitReward,
        weight=1.5,
        params={
            "std": 0.06,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("leg3_link4", "leg2_link4"), ("leg4_link4", "leg1_link4")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    # (11) Keep the coxa (hip yaw) joints close to zero -- i.e. legs close to their symmetric URDF-default
    # splay -- so the policy doesn't twist legs into extreme yaw angles that risk self-collision/twisting up.
    # Weight tightened 0.05 -> 0.15 (was the weakest of the three leg-shape penalties) per feedback: the
    # symmetric "X" stance looks right at spawn but visibly degrades/twists a bit once walking -- coxa isn't
    # hitting its hard limit (coxa_limit termination is <1%), it's drifting within the soft range enough to
    # be noticeable, which is exactly what this term is meant to hold in check, just needed more weight.
    coxa_orthogonal = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg.*_joint1"])},
    )
    # (11b) Keep the femur-end (joint3, the "knee") closer to its default too -- it was never hitting the
    # hard +/-2.0 safety limit, but was swinging wide enough sideways to look off. Same soft-penalty
    # treatment as the coxa: pull it toward center without hard-cutting episodes over it. Weight tripled
    # from -0.05 -> -0.15 per feedback that it still needed tightening further after the first pass.
    femur_end_centered = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg.*_joint3"])},
    )
    # (11c) Keep the tibia (joint4) links pointing straight down while walking, not splayed sideways.
    tibia_vertical = RewTerm(
        func=v1x1_mdp.tibia_vertical_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["leg1_link4", "leg2_link4", "leg3_link4", "leg4_link4"]
            )
        },
    )
    # (11d) Keep the tibia (joint4) angle itself close to its resting default, not just its world-frame
    # orientation (tibia_vertical, above) -- distinct failure mode: a leg can over-EXTEND (joint4 swinging
    # open toward the far end of its range) while technically still hanging roughly vertical. Same
    # soft-centering pattern as coxa_orthogonal/femur_end_centered.
    tibia_extension = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg.*_joint4"])},
    )
    # (12) Keep the chassis level with the ground (not just "not upside down" -- upright's 0.93 threshold is
    # fairly permissive; this gives a smoother, stronger gradient toward actually flat).
    # Weight tightened 0.1 -> 0.2 -> 0.3 per feedback: play-mode viewing showed base_link a degree or two off
    # level while walking, and end-of-episode wobble persisting even after the first tightening pass; this
    # term already penalizes exactly that (projected-gravity XY on the root), just needed more weight.
    # Weight tightened further 0.3 -> 0.5 -> 0.7 -> 1.3 for the fresh-restart lineage per feedback wanting
    # base_link "very parallel to the ground, hardly any tilt at all." Pushed much harder than prior
    # increments here specifically because this is a FRESH restart (iteration 0, not a resume) -- fresh
    # starts have consistently handled aggressive reward weights better than resumed lineages throughout
    # this project, since there's no already-converged policy to fight, just exploration from scratch.
    # Modest further bump -1.3 -> -1.6 per feedback confirming a slight forward tilt is still visible live --
    # this term's raw value has stayed small/stable this whole lineage, so a moderate push (not a big jump,
    # since the tilt was described as only slight) should be enough rather than a large escalation.
    chassis_level = RewTerm(func=mdp.flat_orientation_l2, weight=-1.6)
    # (13) Penalize the chassis facing a different direction than it's actually traveling in -- i.e. veering
    # off its own heading. Self-referential (heading vs. own velocity), NOT the same mechanism as
    # move_to_target (which compares heading against a fixed world target via the hardcoded, structurally
    # wrong-for-this-robot LOCAL +X FORWARD_VEC_B) -- that was already ruled out as the veer's cause, since
    # the from-scratch side-test lineage trained with move_to_target=0.0 for its entire history and still
    # veers. This term uses v1x1's actual anatomical forward axis (LOCAL +Y) instead, so it targets the same
    # observed symptom (left-veering near end of episode) without the old structural mismatch.
    # Weight doubled again 0.3 -> 0.6 -> 1.1 for the fresh-restart lineage per feedback wanting the vector
    # "hardly off at all" -- raw values for this term have stayed small and well-behaved every run so far,
    # and same fresh-start reasoning as chassis_level above applies to pushing it harder here.
    # Weight pushed further -1.1 -> -1.5 per feedback wanting direction alignment tightened even more.
    heading_to_velocity = RewTerm(func=v1x1_mdp.heading_velocity_l2, weight=-1.5, params={"min_speed": 0.15})
    # (14) Nudge base_link a bit higher off the ground per feedback. 0.15m target_height is a starting
    # estimate (this robot's current typical standing height hasn't been precisely measured under the final
    # stance config -- X-splay + joint2 tuck + joint3 nudge all shift it from earlier measurements), not a
    # verified number -- kept the weight modest so it nudges rather than fights the gait if the estimate is
    # off; verify visually and adjust the target once this is live.
    base_height = RewTerm(func=mdp.base_height_l2, weight=-0.3, params={"target_height": 0.15})
    # (15) Reward the front feet for sitting forward of base_link, in the body's own frame -- a direct
    # task-space reward rather than an indirect joint-angle nudge (see mdp.py docstring). Added for the
    # fresh-restart lineage specifically to sidestep the joint-sign-convention fragility that made the
    # previous lineage's front-coxa/joint3 defaults slow to fully take hold.
    # Weight and cap both bumped (0.3 -> 0.4, 0.35 -> 0.45) per feedback wanting more forward reach.
    # Weight and cap both pushed further (0.4 -> 0.6, 0.45 -> 0.6) per feedback specifically wanting the
    # front feet reaching forward more than a general stride increase -- pushed this one harder than the
    # general stride term below since forward reach was singled out as the priority.
    # Modest further nudge (cap 0.6 -> 0.7) per feedback that reach is "almost there" -- small push rather
    # than a big jump since we're close, paired with the new foot_clearance term which may also help via a
    # bigger swing arc.
    front_foot_forward = RewTerm(
        func=v1x1_mdp.front_foot_forward_bonus,
        weight=0.6,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["leg3_link4", "leg4_link4"]), "cap": 0.7},
    )
    # (16) Penalize left/right joint asymmetry between mirrored leg pairs -- a general, direct fix for the
    # recurring "one specific leg behaves differently than its mirror" pattern (front-right knee/tibia sweep
    # across multiple lineages this session) rather than continuing to patch individual joints one at a time.
    left_right_symmetry = RewTerm(func=v1x1_mdp.left_right_symmetry_l2, weight=-0.2)
    # (17) Penalize front/rear leg pairs (same side) using very differently sized joint motions -- see
    # mdp.py docstring for why this compares motion MAGNITUDE rather than raw signed angle (the trot gait's
    # front/rear phase offset makes direct signed comparison meaningless).
    front_rear_symmetry = RewTerm(func=v1x1_mdp.front_rear_symmetry_l2, weight=-0.2)
    # (18) Reward the feet for reaching a target clearance height off the ground during swing (gated by
    # horizontal foot velocity via tanh, so it's inactive while a foot is planted) -- reused unmodified from
    # Isaac Lab's own Spot task (same source as trot_gait), which has no v1x1-specific issue to fix. Per
    # feedback: a bigger swing arc may naturally encourage more forward reach per step, not just longer air
    # time. Spot's own defaults (target_height=0.1, std=0.05) are scaled down for v1x1's much smaller/more
    # compact legs (roughly 1/3-1/2 Spot's scale) -- not empirically measured, verify visually and adjust.
    foot_clearance = RewTerm(
        func=v1x1_mdp.foot_clearance_reward,
        weight=0.3,
        params={
            "target_height": 0.04,
            "std": 0.02,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["leg1_link4", "leg2_link4", "leg3_link4", "leg4_link4"]
            ),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    # NOTE: v1x1's base_link settles at ~0.0m collapsed (measured empirically) vs. ~0.09-0.15m standing
    # (measured via FK search, see robots/v1x1.py init_state). 0.04 sits roughly halfway between those,
    # giving real "fell over" shaping signal instead of just avoiding instant termination.
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.04})
    # (3) Terminate if the robot tips over and gets stuck on its side/back -- without this, a flipped robot
    # can sit collecting the `alive` bonus (weight 0.5, second-highest reward term) indefinitely without any
    # pressure to recover, which actively works against learning to walk. ~115 deg (2.0 rad) from upright
    # still allows normal dynamic tilt during a gait; it only cuts off clear, likely-unrecoverable tip-overs.
    orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 2.0})
    # (4) Terminate if any foot is held continuously airborne too long -- catches "raise a leg and hold it
    # up," which feet_air_time (a reward, not a termination) doesn't actually punish (see mdp.py docstring).
    foot_air_time = DoneTerm(
        func=v1x1_mdp.foot_air_time_exceeded,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg1_link4", "leg2_link4", "leg3_link4", "leg4_link4"]
            ),
            "max_air_time": 1.5,
        },
    )
    # (5) Terminate if any single foot stays planted too long -- catches using one foot as a fixed pivot
    # instead of cycling all four through a gait.
    foot_contact_time = DoneTerm(
        func=v1x1_mdp.foot_contact_time_exceeded,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg1_link4", "leg2_link4", "leg3_link4", "leg4_link4"]
            ),
            "max_contact_time": 3.0,
        },
    )
    # (6) Terminate if all four feet are simultaneously, continuously grounded too long -- catches
    # "stand still in a wide/T-pose stance and never move," which is exactly what we saw converge to.
    all_feet_grounded = DoneTerm(
        func=v1x1_mdp.all_feet_grounded_too_long,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg1_link4", "leg2_link4", "leg3_link4", "leg4_link4"]
            ),
            "max_time": 2.0,
        },
    )
    # (7) Terminate if any coxa (hip yaw) joint swings too far from center -- avoids the policy twisting legs
    # into extreme yaw angles that risk self-collision/twisting up. Tighter than the hardware +/-2.356 limit;
    # this is a training-time safety bound, not a hardware constraint.
    # Bounds tightened +/-1.0 -> +/-0.7 per feedback -- was sitting at 0% firing rate for a long stretch,
    # comfortable room to tighten further (still clear of the +/-0.25 default splay +/- the 0.2 reset jitter).
    coxa_limit = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"bounds": (-0.7, 0.7), "asset_cfg": SceneEntityCfg("robot", joint_names=["leg.*_joint1"])},
    )
    # (8) Terminate if any femur-end (joint3, the "knee") swings too close to its hardware limit -- unlike
    # the coxa, this joint needs real range of motion for stepping clearance, so the bound is looser than
    # coxa's, just cutting off the most extreme/twisted end of its +/-2.356 hardware range.
    # Bounds tightened +/-2.0 -> +/-1.5 per feedback: was so loose (near the full +/-2.356 joint range) it
    # almost never fired, giving no real backstop against the observed front-right knee sweeping too far back.
    # Bounds tightened further +/-1.5 -> +/-1.1 per feedback -- last tightening pass (+/-2.0 -> +/-1.5) still
    # showed 0% firing, so there's room before this actually constrains the policy.
    femur_end_limit = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"bounds": (-1.1, 1.1), "asset_cfg": SceneEntityCfg("robot", joint_names=["leg.*_joint3"])},
    )
    # (9) Terminate if any tibia (joint4) swings too far -- new hard safety bound, complementing the
    # existing tibia_vertical soft-penalty reward, since the tibia was still drifting toward horizontal.
    # Bounds tightened +/-1.8 -> +/-1.4 per feedback, same reasoning as femur_end_limit above.
    tibia_limit = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"bounds": (-1.4, 1.4), "asset_cfg": SceneEntityCfg("robot", joint_names=["leg.*_joint4"])},
    )


@configclass
class V1X1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the v1x1 quadruped walking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=True)
    # Basic settings
    observations: V1X1ObservationsCfg = V1X1ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        # bumped 16.0 -> 20.0 per feedback wanting more time per spawn for the gait to settle/stabilize
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = V1X1PhysicsCfg()
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
