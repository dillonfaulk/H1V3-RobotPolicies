"""Microduck PrecisionStep task — step to a small target pose and stop dead, feet square.

Episodic policy for the last stretch of the BallKick mission (n3z-jr brief,
2026-11-10; Dillon picked the command layout 2026-11-11). The go-to-point
supervisor cruises to ~0.3 m of the ball; this policy closes the rest and
stops with the feet squared, then the kick policy swaps in.

Measured facts it is built on (BallKick mission harness, 2026-11-09):
  - VelStand cannot creep: <= 10 ticks of 0.2 m/s only twists in place, 15
    ticks is a whole 25-38 mm step. A slowed velocity clone would inherit that
    floor, so this is an episodic step-to-pose task instead (BallKick mold).
  - A staggered stance (support foot behind) stops the kick. The terminal
    reward includes foot squareness, so the policy ends in the kick's HOME
    stance by construction.
  - Every remaining mission failure was a bumped ball. The ball is in the
    training scene where the kick needs it relative to the TARGET stance;
    moving it ends the episode.

Command contract ("option A"): the 3 twist slots carry the REMAINING offset to
the target pose in the trunk's current yaw frame, [dx m, dy m, dyaw rad]
(``StepTargetCommand`` in mdp.py). Head/body slots stay zero-padded. The actor
stays ball-blind: the ball's position is implied by the target, never observed.

Robot: the true all-collisions model (as VelStand, and as the mission deploys),
not BallKick's ground-contact model. That avoids BallKick's cross-model seam.

Regularizers, DR and obs noise: BallKick parity (itself velocity parity).
"""

import math
from copy import deepcopy

# Symmetry: OFF. The ball is placed for the RIGHT-foot kick, so ball episodes
# are not mirror-symmetric.
ENABLE_SYMMETRY = False

# ── Domain randomisation (matched to BallKick / velocity) ────────────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

COM_RANDOMIZATION_RANGE             = 0.003
HEAD_COM_RANDOMIZATION_RANGE        = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
VELOCITY_PUSH_RANGE                 = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# ── Task constants ────────────────────────────────────────────────────────────
# 0.25 m at a careful ~0.1 m/s is ~2.5 s, plus a settle window to be paid for
# standing still on target.
EPISODE_LENGTH_S = 6.0

# Target sampling (start yaw frame). Mass at small radii: that is where the
# pulse aligner fails today. Far tail covers the 0.25 m final-approach zone.
TARGET_NEAR_R        = (0.0, 0.08)
TARGET_FAR_R         = (0.08, 0.25)
TARGET_FAR_PROB      = 0.35
TARGET_FWD_CONE_PROB = 0.5        # rest of the bearings are all-around
TARGET_YAW_RANGE     = math.radians(15)
TARGET_ZERO_PROB     = 0.1        # exact "hold still" bucket (AGENTS: zero command must be trained)

# Arrival tolerance (inside the kick's +-15 mm training envelope).
ARRIVE_POS_TOL = 0.012
ARRIVE_YAW_TOL = math.radians(3)

# Ball-as-obstacle (Dillon approved 2026-11-10). BallKick's placement relative
# to the TARGET stance, same +-15 mm noise; a minority of episodes ball-free.
BALL_PROB          = 0.7
BALL_RADIUS        = 0.035
BALL_OFFSET        = (0.09, -0.042)
BALL_POS_NOISE_XY  = 0.015
# Duck's HOME collision footprint (trunk yaw frame, geoms within one ball
# diameter of the soles; measured 2026-11-11). A ball that would overlap it
# at the START pose (expanded by radius + 6 mm: at +3 mm, 3 in 1262 zero-action
# holds tapped a ball spawned 74 mm ahead as the duck leaned) -> ball-free instead.
FOOTPRINT_X        = (-0.033, 0.034)
FOOTPRINT_Y        = (-0.079, 0.079)
BALL_MOVED_FAIL    = 0.01         # m of ball displacement = struck = failed episode

STAND_Z = 0.115
HOME_FOOT_WIDTH = 0.084           # feet centers at (0, +-0.042) at HOME

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
    MICRODUCK_BALL_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_precision_step_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg = make_velocity_env_cfg()

    # Robot MUST stay the first entity (ground-state events write qpos[:, 0:7]).
    cfg.scene.entities = {
        "robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
        "ball":  MICRODUCK_BALL_CFG,
    }
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S
    # VelStand needs 200 on this model; + headroom for ball contacts.
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 220)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity", "track_angular_velocity", "air_time",
        "foot_clearance", "foot_swing_height", "foot_slip", "pose", "soft_landing",
    ]:
        cfg.rewards.pop(name, None)

    # ── Rewards: get to the target ────────────────────────────────────────────
    # Two-scale position/heading Gaussians: the coarse ones give gradient across
    # the zone, the fine ones (std ~ the arrival tolerance) pay for precision.
    cfg.rewards["target_pos_coarse"] = RewardTermCfg(
        func=microduck_mdp.step_target_position, weight=1.0, params={"std": 0.06})
    cfg.rewards["target_pos_fine"] = RewardTermCfg(
        func=microduck_mdp.step_target_position, weight=2.0, params={"std": 0.012})
    cfg.rewards["target_yaw_coarse"] = RewardTermCfg(
        func=microduck_mdp.step_target_heading, weight=0.5, params={"std": math.radians(12)})
    cfg.rewards["target_yaw_fine"] = RewardTermCfg(
        func=microduck_mdp.step_target_heading, weight=1.0, params={"std": math.radians(3)})
    # Potential-based progress: the only gradient for far targets, unfarmable.
    # 5.0 x 0.1 m/s = +0.5/step while approaching, 0 holding, negative retreating.
    cfg.rewards["target_progress"] = RewardTermCfg(
        func=microduck_mdp.step_target_progress, weight=5.0, params={"max_speed": 0.3})
    # Arrive and STOP DEAD (both feet down, trunk still, inside tolerance).
    cfg.rewards["target_settled"] = RewardTermCfg(
        func=microduck_mdp.step_target_settled, weight=3.0,
        params={"sensor_name": feet_ground_cfg.name, "pos_tol": ARRIVE_POS_TOL, "yaw_tol": ARRIVE_YAW_TOL})

    # ── Penalties (self-negating functions -> POSITIVE weights; all log <= 0) ─
    # Feet square near the target: 20 mm stagger costs -0.6/step at weight 30.
    cfg.rewards["foot_stagger"] = RewardTermCfg(
        func=microduck_mdp.step_foot_stagger_penalty, weight=30.0,
        params={"near": 0.04, "home_width": HOME_FOOT_WIDTH,
                "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))})
    # Don't step THROUGH the target: 20 mm past (beyond a 10 mm margin) = -0.2/step.
    cfg.rewards["overshoot"] = RewardTermCfg(
        func=microduck_mdp.step_overshoot_penalty, weight=20.0, params={"margin": 0.01})
    # Struck the ball: one-off -20, and the episode ends (forfeits the rest).
    cfg.rewards["ball_struck"] = RewardTermCfg(
        func=microduck_mdp.ball_displaced_penalty, weight=20.0,
        params={"threshold": BALL_MOVED_FAIL, "asset_name": "ball"})

    # Arrive in the KICK's start stance: legs at HOME once within 30 mm. Added
    # after run 1 (model_1000 parked in a ~30-36 deg hip/knee crouch on target;
    # the kick never swung from it, 0/60 missions). L1 pulls out of the crouch
    # (16 deg mean leg error ~ -1.4/step at weight 5), the std-0.1 Gaussian pays
    # for actually landing within the kick's +-0.05 rad training noise.
    cfg.rewards["arrive_pose_match"] = RewardTermCfg(
        func=microduck_mdp.step_terminal_pose_match, weight=2.0,
        params={"std": 0.1, "joint_indices": _LEG_JOINTS, "near": 0.03})
    cfg.rewards["arrive_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.step_terminal_pose_penalty, weight=5.0,
        params={"joint_indices": _LEG_JOINTS, "near": 0.03})

    # ── Rewards: stand cleanly (BallKick stack, legs looser: stepping) ────────
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match, weight=1.0,
        params={"std": 0.5, "joint_indices": _LEG_JOINTS, "target_overrides": None})
    cfg.rewards["pose_stand_neck"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match, weight=1.0,
        params={"std": 0.3, "joint_indices": _NECK_JOINTS, "target_overrides": None})
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian, weight=1.0,
        params={"std": 0.04, "target_height": STAND_Z,
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))})

    # ── Sim2real regularisers — BallKick parity ──────────────────────────────
    cfg.rewards["action_rate_l2"].weight = -0.1  # curriculum ramps to -1.0
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": self_collision_cfg.name})

    # ── Observations (unified 61D actor layout, ball-blind) ───────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel, scale=1.0)
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    g = "projected_gravity"
    cfg.observations["actor"].terms[g] = deepcopy(cfg.observations["actor"].terms[g])
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(cfg.observations["actor"].terms["base_ang_vel"])
    for term in ("base_ang_vel", g):
        cfg.observations["actor"].terms[term].delay_min_lag = 0
        cfg.observations["actor"].terms[term].delay_max_lag = 1
        cfg.observations["actor"].terms[term].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[g].noise              = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise    = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise    = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        gv = cfg.observations["actor"].terms[g]
        gv.func = microduck_mdp.projected_gravity_imu_misaligned
        gv.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(cfg.observations["actor"].terms["joint_vel"])
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4})
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6})

    # Critic-only ball state (parked balls read as ~3 m away = "no ball").
    cfg.observations["critic"].terms["ball_position"] = ObservationTermCfg(
        func=microduck_mdp.ball_pos_in_base, params={"asset_name": "ball"})
    cfg.observations["critic"].terms["ball_velocity"] = ObservationTermCfg(
        func=microduck_mdp.ball_vel_in_base, params={"asset_name": "ball"})

    # ── Command: remaining offset to the target, in the twist slots ───────────
    # One fixed target per episode (resampling period > episode length).
    cfg.commands["twist"] = microduck_mdp.StepTargetCommandCfg(
        resampling_time_range=(EPISODE_LENGTH_S * 10, EPISODE_LENGTH_S * 10))

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations["nan_state"] = TerminationTermCfg(func=microduck_mdp.robot_state_is_nan, time_out=False)
    cfg.terminations["ball_struck"] = TerminationTermCfg(
        func=microduck_mdp.ball_displaced_termination, time_out=False,
        params={"threshold": BALL_MOVED_FAIL, "asset_name": "ball"})

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields, mode="startup")
    cfg.events["reset_action_history"] = EventTermCfg(func=microduck_mdp.reset_action_history, mode="reset")
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = ("left_foot_collision", "right_foot_collision")
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)
    # Start-stance noise: the handoff from VelStand never lands exactly on HOME
    # (BallKick's +-0.05 was too narrow — its later checkpoints overfit HOME).
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.08, 0.08)
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state, mode="reset",
        params={
            "face_down_prob": 0.0, "face_up_prob": 0.0, "sitting_prob": 0.0, "standing_prob": 1.0,
            "sitting_tilt_max": math.radians(5), "standing_z_min": 0.11, "standing_z_max": 0.12,
        })
    # MUST come after set_ground_state (reads the final robot pose).
    cfg.events["reset_step_target"] = EventTermCfg(
        func=microduck_mdp.reset_step_target, mode="reset",
        params={
            "near_r": TARGET_NEAR_R, "far_r": TARGET_FAR_R, "far_prob": TARGET_FAR_PROB,
            "fwd_cone_prob": TARGET_FWD_CONE_PROB, "yaw_range": TARGET_YAW_RANGE,
            "zero_prob": TARGET_ZERO_PROB, "ball_prob": BALL_PROB, "ball_offset": BALL_OFFSET,
            "ball_noise_xy": BALL_POS_NOISE_XY, "ball_radius": BALL_RADIUS,
            "keepout_x": FOOTPRINT_X, "keepout_y": FOOTPRINT_Y, "asset_name": "ball",
        })

    if ENABLE_VELOCITY_PUSHES:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity, mode="interval",
            interval_range_s=(0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S,
            params={"velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                    "asset_cfg": SceneEntityCfg("robot")})
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "operation": "add",
                    "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE)})
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES), "operation": "add",
                    "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE)})
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)), "operation": "scale",
                    "ranges": ARMATURE_RANDOMIZATION_RANGE})
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        lo, hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                    "alpha_range": (math.log(lo) / 2.0, math.log(hi) / 2.0)})
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE})

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum (BallKick's stages) ────────────────────────────────────────
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={"reward_name": "action_rate_l2", "weight_stages": [
            {"step": 0, "weight": -0.1}, {"step": 500 * 24, "weight": -0.2},
            {"step": 750 * 24, "weight": -0.4}, {"step": 1000 * 24, "weight": -0.6},
            {"step": 1250 * 24, "weight": -0.8}, {"step": 1500 * 24, "weight": -1.0}]})
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={"event_name": "randomize_com", "range_stages": [
                {"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005},
                {"step": 1000 * 24, "range": 0.01}, {"step": 1500 * 24, "range": 0.015}]})
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={"event_name": "randomize_head_com", "range_stages": [
                {"step": 0, "range": 0.003}, {"step": 500 * 24, "range": 0.005},
                {"step": 1000 * 24, "range": 0.01}]})
    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={"event_name": "push_robot", "push_stages": [
                {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 500 * 24, "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                {"step": 1000 * 24, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}}]})

    return cfg


MicroduckPrecisionStepRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.01,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
        gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="precision_step",
    run_name="precision_step",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=4000,
)
