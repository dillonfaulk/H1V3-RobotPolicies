# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the asimov-1 open-source humanoid robot (menloresearch/asimov-1)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

ASIMOV1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/n3z/IsaacLab/source/isaaclab_assets/data/Robots/asimov1/asimov_1/asimov_1.usda",
        # required for ContactSensor-based terminations (falling/torso contact) to see anything at all --
        # PhysX bodies don't report contacts by default, this is an explicit opt-in per body.
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # enabled live mid-session: user observed legs visually clipping/phasing through each other
            # (self-collision off, mirroring H1_CFG's convention, which was never actually right for THIS
            # robot -- same fix v1x1's own compact legs needed early on for the identical symptom).
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        copy_from_source=False,
    ),
    # all-zero joint pose is already a straight-leg standing stance for this robot (the knee joint's own
    # limits are asymmetric by design -- left_knee: [0, 1.5], right_knee: [-1.5, 0] -- so 0 == straight leg,
    # unlike v1x1 where the all-zero pose was a degenerate non-standing shape needing hand-derived offsets).
    # pos.z=0.63 is derived by summing the leg joint chain's z-offsets at zero angle (pelvis -> hip_pitch ->
    # hip_roll -> hip_yaw -> knee -> ankle_pitch -> ankle_roll -> foot sole) -- an ESTIMATE pending visual
    # confirmation, same caveat discipline as every v1x1 placeholder.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.63),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # per-joint effort_limit_sim values taken directly from the source URDF's own effort="..." specs
        # (same approach v1x1 used). stiffness/damping PD gains have NO real-hardware basis (the real
        # actuators are firmware-PD-controlled on real current/torque, not sim-exposed) -- these are
        # ESTIMATES, scaled down proportionally from Isaac Lab's own H1/G1 humanoid gains relative to their
        # much higher effort limits, flagged for tuning after the first visual smoke test.
        # stiffness REBALANCED live mid-session (FRESH3 restart): direct comparison against Unitree
        # H1_MINIMAL_CFG's own real gains (isaaclab_assets/robots/unitree.py) found our legs/waist running
        # at a stiffness/effort_limit ratio 2-4x HIGHER than H1's (H1 hip/knee: 150-200/300 = 0.5-0.667;
        # ours was 80/45=1.78 on hip_pitch/roll/knee, 80/28=2.86 on hip_yaw). A PD controller this stiff
        # relative to its torque ceiling saturates (clips) for LARGE position-offset commands sooner than a
        # properly-proportioned one -- plausible direct mechanism for the forward-specific deficit, since a
        # confident forward stride needs the biggest/fastest joint excursions of any commanded direction
        # (small backward/sideways catch-steps rarely approach saturation). Compounds with the earlier
        # action_scale 0.25->0.5 raise (bigger commanded offsets = more torque demanded = more saturation),
        # which was never re-checked against gains afterward. Retargeted to H1's own ratio (~0.6, middle of
        # its 0.5-0.667 range) scaled to OUR effort limits: hip_pitch/roll/knee 45*0.6=27, hip_yaw 28*0.6β‰ˆ17.
        # damping left unchanged (2.0) -- H1 itself uses a FLAT damping (5.0) despite stiffness varying
        # 150-200, so preserving a strict stiffness/damping ratio isn't what H1's own recipe actually does.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint"],
            effort_limit_sim={
                ".*_hip_pitch_joint": 45.0,
                ".*_hip_roll_joint": 45.0,
                ".*_hip_yaw_joint": 28.0,
                ".*_knee_joint": 45.0,
            },
            stiffness={
                ".*_hip_pitch_joint": 27.0,
                ".*_hip_roll_joint": 27.0,
                ".*_hip_yaw_joint": 17.0,
                ".*_knee_joint": 27.0,
            },
            damping=2.0,
        ),
        # ankle_pitch was the OPPOSITE problem -- proportionally WEAKER than H1's ankle (ours 20/40=0.5 vs
        # H1's flat ankle ratio 20/20=1.0), which matters for push-off propulsion. Retargeted to H1's ratio
        # (1.0): ankle_pitch 40*1.0=40, ankle_roll 17*1.0=17. damping raised 1.0->2.0 to match H1's own flat
        # ankle damping exactly (was running at half of H1's).
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 17.0,
            },
            stiffness={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 17.0,
            },
            damping=2.0,
        ),
        # same over-stiffness correction as legs above -- H1's torso_joint ratio is 200/300=0.667; ours was
        # 60/40=1.5, more than 2x over. Retargeted to 40*0.667β‰ˆ27. damping left unchanged (1.5).
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=40.0,
            stiffness=27.0,
            damping=1.5,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 30.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*_elbow_joint": 12.0,
                ".*_wrist_yaw_joint": 12.0,
            },
            stiffness=40.0,
            damping=1.0,
        ),
    },
)
"""Configuration for the asimov-1 humanoid robot (23 actuated DOF: 6/leg x2, 1 waist yaw, 5/arm x2)."""
