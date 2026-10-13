# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.physics import KaminoSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from . import mdp as asimov1_mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.asimov1 import ASIMOV1_CFG  # isort: skip


@configclass
class Asimov1PhysicsCfg(PresetCfg):
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
class Asimov1ContactSensorCfg(PresetCfg):
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
    """Configuration for the terrain scene with an asimov-1 robot."""

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
        # purely cosmetic -- visual_material only affects rendering, collision geometry/friction stay
        # governed by physics_material above, untouched. Per user request ("spruce him up... any type of
        # ground will do, just not a boring grid"). IMPORTANT MECHANISM FOUND (read source before guessing):
        # for terrain_type="plane" specifically, `TerrainImporterCfg.import_ground_plane` does NOT actually
        # bind visual_material as a real texture at all -- it only tries to read a flat `diffuse_color` key
        # out of `visual_material.to_dict()` to tint `GroundPlaneCfg`'s own fixed grid-USD asset
        # (`{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd`, hardcoded, not swappable via
        # this cfg). Real repeating image texturing (what H1/G1 use `MdlFileCfg` for) only works on
        # terrain_type="generator" (a totally different code path, procedural terrain mesh) -- NOT applied
        # here deliberately, since switching terrain_type would also change how env origins get computed
        # (terrain-grid-derived vs. scene-cloning-derived), a real risk to training behavior mid-curriculum
        # that isn't worth a cosmetic swap. Tried MdlFileCfg twice (vMaterials_2/Stone/Steel_Grey, then even
        # H1/G1's own exact TilesMarbleSpiderWhiteBrickBondHoned) -- BOTH silently fell back to solid BLACK
        # because MdlFileCfg's own `.to_dict()` doesn't expose a `diffuse_color` key at all (caught via
        # isolation test both times, never reached a real run). Fixed by using `PreviewSurfaceCfg` instead,
        # which DOES expose `diffuse_color` directly -- gets an intentional solid color through the only
        # mechanism this terrain_type actually supports, instead of the accidental black default. Muted
        # grass-green chosen since the user's own first suggestion was grass (no real grass texture exists
        # in this asset bucket, confirmed by listing every Materials/Environments subdirectory).
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.85, 0.0)),
        debug_vis=False,
    )

    # robot
    robot = ASIMOV1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    contact_forces = Asimov1ContactSensorCfg()

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    # sky backdrop -- purely cosmetic (per user request, "make the backdrop sky-blue... the background sky
    # area above the horizon"). A solid-color DomeLightCfg (no texture_file) renders as a visible flat-color
    # sky dome by default (`visible_in_primary_ray=True`), while ALSO providing soft ambient fill light
    # alongside the existing DistantLightCfg's directional "sun" shadows -- standard sun+sky light pairing.
    # Zero physics/training impact, same category of change as the ground/robot color edits.
    sky = AssetBaseCfg(
        prim_path="/World/sky",
        spawn=sim_utils.DomeLightCfg(color=(0.4, 0.65, 0.95), intensity=750.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # position control (NOT effort/torque control, unlike v1x1) -- mirrors Isaac Lab's own Unitree H1/G1
    # humanoid tasks. Balance/standing is dramatically easier to bootstrap under PD/position control than
    # raw torque control, which is why H1/G1 don't use JointEffortActionCfg the way v1x1 does -- v1x1's
    # torque-control choice was specific to continuing that project's own established gait-tuning history,
    # not a universal rule for every robot.
    # scale lowered 0.5 -> 0.25 live mid-session: at scale=0.5, a raw untrained policy's random Gaussian
    # output can command a position target far enough outside the joint's real range that the PD controller
    # (hip velocity limit 12.57 rad/s) swings it into the new hip_pitch_limit termination bound within a
    # handful of physics steps -- confirmed via mean episode length collapsing to ~6 steps with
    # hip_pitch_limit firing 100% of episodes, far too fast to be a learned behavior, just exploration noise
    # amplified by an overly generous action scale.
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # no generated_commands term -- this is a standing-balance task, not a commanded-velocity task
        # (see RewardsCfg below). The policy's only job is "stay upright near your nominal pose."
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class Asimov1ObservationsCfg(PresetCfg):
    default: ObservationsCfg = ObservationsCfg()
    physx: ObservationsCfg = ObservationsCfg()
    newton_mjwarp: ObservationsCfg = ObservationsCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # small pose/velocity jitter at reset -- keeps the policy robust to not spawning in a perfectly idealized
    # state, same spirit as every other task in this project.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
        },
    )

    # exact default pose on reset (position_range=(1.0, 1.0), no random joint scaling), matching Isaac Lab's
    # own H1/G1 choice -- we want a clean, repeatable starting stance to learn to hold rather than adding
    # joint-angle randomization on top of a task that's already hard enough from scratch.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )

    # the actual "balance" training signal -- random lateral push disturbances, Isaac Lab's standard
    # mechanism for "robust to being pushed" (used unmodified by both H1 and G1). Starting GENTLER than
    # H1/G1's +/-0.5 m/s default since this is a from-scratch standing policy, not a mature walker -- raise
    # it once basic standing is solid.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (1b) Added live mid-session -- CRITICAL fix, not optional polish. Without this, ending an episode
    # early costs the policy nothing beyond forfeiting future `alive` bonus; since early flailing produces
    # large negative per-step penalties from the unbounded L2 terms below (unlike H1/G1's own reward shape,
    # which relies on bounded-positive tracking rewards as its dominant signal and barely has any large
    # unbounded penalties), the value function can conclude "ending this now beats continuing while
    # off-balance." Confirmed happening live: feet_crossed was firing on 100% of episodes, 0% everything
    # else -- the policy had found crossing its feet is the cheapest, fastest way to end an episode on
    # purpose, not an instability symptom. Mirrors H1/G1's own `termination_penalty` term exactly (same
    # function, same -200 weight) -- closes the loophole by making every non-timeout termination cost far
    # more than any amount of continued flailing ever could.
    # Weight raised 200 -> 1000 live mid-session: -200 wasn't enough -- feet_crossed came back to 100% of
    # episodes a second time (via the RIGHT foot this time, confirming it's a general "end this episode
    # fast" strategy, not a left-leg-specific quirk that feet_separation alone could patch). -200 is only
    # worth ~17% of a full 1200-step episode's potential alive-bonus upside -- too small a deterrent against
    # ending early when the value function hasn't yet learned balancing pays off over the long run.
    # Weight raised 1000 -> 1500 per explicit request to tighten up the fall rate further -- run14/run15
    # plateaued around 64% time_out/~36% falling with -1000, suggesting there's still room to push the
    # deterrent harder before diminishing returns set in.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-1500.0)
    # (2) Stay upright -- the core "balance" signal
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    # (3) Stay near the standing height estimate (see robots/asimov1.py init_state for how 0.63 was derived)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-1.0, params={"target_height": 0.63})
    # (4) Don't drift horizontally or rotate -- the two custom terms this task actually needs (see mdp.py
    # docstring for why no stock equivalent exists for a standing-still task).
    lin_vel_xy = RewTerm(func=asimov1_mdp.lin_vel_xy_l2, weight=-1.0)
    ang_vel = RewTerm(func=asimov1_mdp.ang_vel_l2, weight=-0.5)
    # (4b) Added live mid-session, preventive: a separate, more targeted penalty on just the YAW-axis
    # joints (hip_yaw, waist_yaw) -- `joint_deviation_legs`/`joint_deviation_waist` below already cover
    # these but lump them in with pitch/roll/knee, diluting the signal. Queued per explicit request to
    # "stop a hip/torso rotation habit if it starts to develop" -- not yet observed live, added
    # preemptively alongside the termination_penalty fix above.
    # Weight tightened -0.5 -> -1.5 per explicit feedback after run12's strong result.
    hip_torso_rotation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", "waist_yaw_joint"])},
    )
    # (4d) Added live mid-session per explicit request, mirroring v1x1's own HeadingHoldBonus: captures the
    # world-frame yaw heading at the start of each episode and rewards staying close to it over time -- a
    # genuine restoring force against accumulated drift that `ang_vel`/`hip_torso_rotation` (current rate
    # and joint-angle deviation only, no memory of the episode's starting heading) can't provide alone.
    heading_hold_bonus = RewTerm(func=asimov1_mdp.HeadingHoldBonus, weight=2.0, params={"asset_cfg": SceneEntityCfg("robot")})
    # (4b2) Added live per explicit request: "give a hint to step to catch themselves if they lean too far."
    # Only pays out in the overlap of "leaning" (gated, see mdp.py docstring) and "a foot is currently
    # airborne" -- a lightweight nudge toward a genuine recovery step, not a blanket stepping incentive.
    recovery_step_bonus = RewTerm(
        func=asimov1_mdp.recovery_step_bonus,
        weight=2.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "tilt_threshold": 0.15,
            "tilt_range": 0.35,
        },
    )
    # (4c) Added live mid-session -- the real fix for the feet-crossing exploit. termination_penalty (above)
    # raised the cost of triggering feet_crossed but didn't remove the exploit, just delayed its rediscovery
    # (confirmed live: feet_crossed went back to firing on 100% of episodes a bit later in training). The
    # actual problem is that feet_crossed is a binary cliff-edge signal with zero gradient as the legs
    # approach each other -- this term gives continuous, graded pushback the whole way there instead.
    # min_separation=0.1m sits just inside the ~0.135m natural standing width (hip offsets +/-0.0675m), so
    # normal stance incurs no penalty, only genuine inward drift does.
    feet_separation = RewTerm(
        func=asimov1_mdp.feet_separation_l2,
        weight=-50.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "min_separation": 0.1,
        },
    )
    # (4e) Added live mid-session per explicit request: a symmetric attractor toward the robot's own natural
    # resting stance width (0.215m, computed from the leg chain's zero-pose geometry -- see mdp.py docstring)
    # rather than just avoiding the narrow/wide extremes, which previously left a wide unrewarded dead zone.
    feet_stance_width = RewTerm(
        func=asimov1_mdp.feet_stance_width_l2,
        weight=-20.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "target_separation": 0.215,
        },
    )
    # (5) Stay near the nominal (all-zero) joint pose, grouped by limb region -- same pattern H1/G1 use.
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    joint_deviation_feet = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_yaw_joint"]
            )
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.2, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])}
    )
    # (5b) Added live mid-session: policy was observed persistently kicking one leg out to the side instead
    # of learning to stand. Penalizes the shin links tilting away from vertical in the world frame -- same
    # mechanism as v1x1's own tibia_vertical_l2 fix for an analogous "leg splaying sideways" symptom.
    # Confirmed working live: the leg-kick behavior disappeared after adding this.
    shin_vertical = RewTerm(
        func=asimov1_mdp.body_vertical_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_knee_link", "right_knee_link"])},
    )
    # (5c) Added right after shin_vertical confirmed working -- flat_orientation (above) only checks the
    # pelvis/root, not this separate upper-body chest segment, which could tilt independently. Weight kept
    # nearly equal to shin_vertical per explicit feedback.
    torso_vertical = RewTerm(
        func=asimov1_mdp.body_vertical_l2,
        weight=-0.9,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["waist_yaw_link"])},
    )
    # (6) Smoothness penalties -- same set H1/G1 use.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls over (pelvis touching the ground)
    pelvis_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis_link"]), "threshold": 1.0},
    )
    # (3) Terminate if it collapses (standing height ~0.63m estimate, collapsed is near 0m -- 0.3 sits
    # roughly halfway, same shaping-margin reasoning used for v1x1's torso_height termination)
    collapsed = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.3})
    # (4) Terminate if it tips past a recoverable angle
    orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    # (5-7) Added live mid-session: policy was observed kicking a hip joint to an extreme angle (leg flying
    # up and behind), driving 100% collapsed terminations. Hard bounds well inside the hardware limits
    # (hip_pitch [-1.57,1.57], hip_roll/_yaw [-0.785,0.785]) as a training-time safety backstop, mirroring
    # v1x1's coxa_limit/femur_end_limit/tibia_limit pattern for an analogous extreme-swing problem.
    # Bounds loosened immediately after (0.8->1.2, 0.5->0.65) + action scale lowered (see ActionsCfg) once
    # these first bounds turned out FAR too tight -- they were firing on 100% of episodes within ~6 steps of
    # reset, which is pure untrained exploration noise, not a learned habit (nowhere near enough steps for
    # that). These looser bounds still catch a genuine extreme kick while giving real room for normal
    # balance-correction swings and early random noise to NOT immediately end the episode.
    hip_pitch_limit = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"bounds": (-1.2, 1.2), "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint"])},
    )
    hip_roll_limit = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"bounds": (-0.65, 0.65), "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )
    hip_yaw_limit = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"bounds": (-0.65, 0.65), "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )
    # (8) Added live mid-session: this is a standing task, so sustained loss of foot contact (hopping,
    # one foot flailing in the air) is always a sign of instability, not a valid strategy -- unlike v1x1's
    # walking task, there's no gait cycle here that legitimately needs air time.
    foot_airborne = DoneTerm(
        func=asimov1_mdp.foot_airborne_exceeded,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "max_air_time": 0.4,
        },
    )
    # (9) Added live mid-session: catches feet crossing laterally (left foot ending up on the right side or
    # vice versa) -- a real anatomical/collision concern for a biped, and a sign of a degenerate balance
    # strategy. Body order MUST be [left, right] -- see mdp.py docstring.
    feet_crossed = DoneTerm(
        func=asimov1_mdp.feet_crossed,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                # SceneEntityCfg.preserve_order defaults to False (resolves to the articulation's own
                # internal body index order, NOT necessarily list order) -- must be True here since
                # feet_crossed's math depends on index 0 == left foot, index 1 == right foot specifically.
                preserve_order=True,
            )
        },
    )
    # (10) Added live mid-session: once feet-crossing stopped being viable, some envs started widening their
    # stance well past normal balance-correction width instead.
    # Bound loosened 0.22 -> 0.32 live mid-session: 0.22 was WRONG -- it assumed shoulder width (0.193m) was
    # near the robot's natural resting stance, but the actual all-zero-pose foot separation is 0.215m (the
    # leg chain's own Y-offsets sum wider than the shoulders, not narrower -- see mdp.py's feet_stance_width_l2
    # docstring for the full derivation), so 0.22 left only 5mm of margin above PASSIVE standing with no
    # policy input at all. Confirmed live: fired on 65% of episodes immediately on resume. 0.32m gives real
    # headroom above the 0.215m natural width for genuine recovery-stance widening, while
    # `feet_stance_width` (below) actively pulls back toward 0.215m as the preferred resting width.
    feet_too_wide = DoneTerm(
        func=asimov1_mdp.feet_too_wide,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "max_separation": 0.32,
        },
    )


@configclass
class ASIMOV1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the asimov-1 humanoid stand-and-balance environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=128, env_spacing=3.0, clone_in_fabric=True)
    # Basic settings
    observations: Asimov1ObservationsCfg = Asimov1ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
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
        self.sim.physics = Asimov1PhysicsCfg()
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
