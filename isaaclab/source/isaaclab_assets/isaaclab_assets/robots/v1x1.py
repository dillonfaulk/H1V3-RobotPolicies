# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the custom v1x1 quadruped robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

V1X1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/n3z/IsaacLab/source/isaaclab_assets/data/Robots/v1x1/v1x1_pruned/v1x1_pruned.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        # required for the ContactSensor (feet_air_time / base_contact rewards) to see anything at all --
        # PhysX bodies don't report contacts by default, this is an explicit opt-in per body.
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # unlike Ant (whose sprawled legs never approach the torso), v1x1's compact torque-driven legs
            # were visibly clipping through base_link with self-collision off -- enable it so the legs
            # physically can't pass through the body or each other.
            enabled_self_collisions=True,
            # Ant's solver_position_iteration_count=4 is fine for Ant (its legs never generate meaningful
            # self-contact force); v1x1's torque-driven legs folding toward the body need far more solver
            # iterations to actually resolve contact stiffly enough to stop visible tunneling/interpenetration.
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # v1x1's designed neutral/standing pose is all joints at their URDF zero position (matching how the
        # robot is authored in v1x1_description/urdf/v1x1.urdf.xacro) -- not a hand-engineered FK stance --
        # EXCEPT the coxa (joint1) yaw offsets and the joint2 offset below. Every leg's coxa origin has
        # rpy="0 0 0" (no per-corner pre-rotation) and identical joint2/3/4 offsets, so at all-zero the legs
        # all extend the same direction relative to their own corner (straight out to the side, ~0 front/back
        # component) -- an "I" shape, not a splayed "X" stance. Per feedback, opening front legs forward and
        # rear legs backward via joint1 yaw fixes this. Signs derived from the URDF's actual mount offsets
        # (leg1/leg3 extend toward -X, leg2/leg4 toward +X at zero pose; rotating that direction vector by
        # +theta about +Z swings a -X-pointing leg toward -Y and a +X-pointing leg toward +Y) -- diagonal
        # pairs (leg1+leg4, leg2+leg3) share sign. Front legs (leg3/leg4) bumped further, 0.25 -> 0.45, per
        # follow-up feedback wanting them more forward (not fully centered, just further than the initial
        # pass) -- rear legs left at the original 0.25 since only the front was asked about.
        #
        # joint2 (femur-base/hip pitch, axis Y) is geometrically IDENTICAL across all 4 legs (same origin,
        # same axis, no per-corner rotation) unlike joint1's mount position -- so a single uniform offset
        # (same value, same sign, no left/right mirroring needed) pulls all 4 feet in laterally, tucking them
        # under the body rather than straight out to the side. -0.25 rad is a starting estimate derived from
        # the joint3/joint4/foot offset chain's rotation about joint2's Y axis, not empirically verified yet.
        joint_pos={
            "leg1_joint1": 0.25,  # rear-left -> swings further rearward (-Y)
            "leg2_joint1": -0.25,  # rear-right -> swings further rearward (-Y)
            # REVERTED 0.65 -> 0.45 on the old lineage: 0.65 left only 0.05 rad of margin against the
            # coxa_limit hard bound (then +/-0.7), causing it to fire on 61% of episodes and tanking reward
            # 58 -> 24. This fresh lineage's coxa_limit is also +/-0.7, so bumping more conservatively this
            # time -- 0.45 -> 0.55 leaves 0.15 rad margin (3x the margin that caused the earlier disaster).
            "leg3_joint1": -0.55,  # front-left -> swings further forward (+Y)
            "leg4_joint1": 0.55,  # front-right -> swings further forward (+Y)
            "leg.*_joint2": -0.25,  # all 4 legs -> tucks feet in laterally, under the body
            # front legs only -- nudges joint3 (the "knee," a roll about local X) to bias the tibia toward
            # swinging forward rather than trailing back, addressing an observed front-tibia-sweep quirk.
            # CORRECTED: initially guessed mirrored left/right (like joint1), but that was wrong -- live
            # verification showed leg3 (+0.2) fixed it while leg4 (-0.2) did not. Re-derived: joint3 rotates
            # about local X, so it only changes the downstream chain's Y/Z components; the foot offset that
            # drives the Y-push (0, 0, -0.242) is IDENTICAL for every leg (no left/right sign difference,
            # unlike joint1's mount offset), so the correct relationship is UNIFORM sign, same as joint2, not
            # mirrored. Both front legs get +0.2.
            # Bumped further 0.2 -> 0.35 per feedback wanting more forward reach via the knee too -- huge
            # margin available here (femur_end_limit is +/-1.1, so 0.35 still leaves 0.75 rad of room).
            "leg3_joint3": 0.35,  # front-left
            "leg4_joint3": 0.35,  # front-right
        },
        # Same convention as Ant: start near-neutral, spawn with a modest drop, and let the policy learn to
        # actively hold itself up via torque control rather than pre-solving a "standing" pose for it.
        pos=(0.0, 0.0, 0.5),
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.1,
            armature=0.05,
            effort_limit_sim=100.0,
        ),
    },
)
"""Configuration for the custom v1x1 quadruped robot (4 legs x 4 joints = 16 DOF)."""
