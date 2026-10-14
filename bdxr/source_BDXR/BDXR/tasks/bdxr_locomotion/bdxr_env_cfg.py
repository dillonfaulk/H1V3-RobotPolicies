# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.sensors import ImuCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
#from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv  # <<<<<<<<<< MOVE THE IMPORT HERE
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    # It's fine to keep this import here for type checkers if needed, but the one above is essential for runtime.
    from isaaclab.managers import RewardTermCfg

# TODO: Clean up all non used imports

##
# Pre-defined configs
##

from BDXR.robots.bdxr import BDX_R_CFG # isort:skip

import BDXR.tasks.bdxr_locomotion.mdp as mdp  # isort:skip

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,  # green arrow = commanded velocity, blue arrow = actual velocity, both above the base
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class CriticCfg(ObsGroup):
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        actions = ObsTerm(func=mdp.last_action)
        imu_projected_gravity = ObsTerm(
            # mdp.imu_projected_gravity doesn't exist in this Isaac Lab pin (v3.0.0-beta2.patch1) --
            # ImuData here only exposes ang_vel_b/lin_acc_b, no orientation. Substituting the robot's
            # own root-frame projected_gravity is exact here since this task's IMU offset rotation is
            # identity (ImuCfg.OffsetCfg(rot=(1,0,0,0))), so IMU frame == base_link frame.
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        joint_torques = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # feet_contact_forces (mdp.body_incoming_wrench) dropped: in this Isaac Lab pin that function
        # reads from a dedicated JointWrenchSensor, not an asset_cfg on the Articulation directly -- a
        # real API gap, not a rename. This term is critic-only (privileged info, not part of the policy's
        # own observation space), so dropping it doesn't change the deployed policy's input at all.
        # Joint positions and velocities with less noise (privileged accurate state)
        joint_pos_accurate = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )
        joint_vel_accurate = ObsTerm(
            func=mdp.joint_vel_rel,
            scale =0.05,
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )

        # No noise for the critic
        def __post_init__(self):
            self.enable_corruption = False

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel, scale=0.2, noise=Unoise(n_min=-0.35, n_max=0.35))

        # same substitution as the critic group above -- see that comment for why this is exact, not approximate
        imu_projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation groups:
    critic: CriticCfg = CriticCfg()
    policy: PolicyCfg = PolicyCfg()





@configclass
class BDXRRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    air_time = RewTerm(
        func=mdp.bipedal_air_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot"),
        },
    )
    # penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_Ankle")},
    )
    # penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw", ".*_Hip_Roll"])},
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=2,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot"),
        },
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_Foot"),
            "threshold": 1.0,
        },
    )
    base_height_deviation = RewTerm(
        func=mdp.base_height_l2,
        weight=-2,  # Tune this weight as needed
        params={
            "target_height": 0.30846,
            "asset_cfg": SceneEntityCfg(name="robot", body_names=["base_link"]),
        },
    )
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )





@configclass
class EventCfg:
    """Configuration for events."""
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.2, 0.2),
            "operation": "add",
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
#
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )

@configclass
class BdxrEnvCfg(LocomotionVelocityRoughEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    rewards: BDXRRewards = BDXRRewards()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = BDX_R_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # NOTE: this URDF's on-the-fly conversion (UrdfFileCfg, no --merge-joints-into-root option used)
        # nests the real rigid body/articulation root one level deeper than a plain "<Robot>/base_link"
        # path -- confirmed via direct USD stage probe: the actual RigidBodyAPI+ArticulationRootAPI prim
        # is at ".../Robot/Geometry/base_link", not ".../Robot/base_link". contact_forces (base class
        # default) is unaffected since it uses a ".*" glob, but these literal paths need the extra segment.
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/base_link"

        # CORRECT IMU CONFIGURATION
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
            debug_vis=False,
            offset=ImuCfg.OffsetCfg(  
                pos=(-0.04718, 0.0663, 0.11094),    
                rot=(1.0, 0.0, 0.0, 0.0),           
            )
        )
# actions
        self.actions.joint_pos.scale = 0.5

        # events
        self.events.push_robot.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
        #self.events.push_robot = None
        self.rewards.feet_air_time = None 
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 0.5)
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.physics_material.params["static_friction_range"] = (0.1, 2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.1, 2)
        self.events.physics_material.params["asset_cfg"].body_names = ".*_Foot"
        # Switched from the stock multi-row/col rough-terrain generator (COBBLESTONE_ROAD_CFG, every cell
        # literally flat anyway -- "cobblestone" was a misnomer, just one MeshPlaneTerrainCfg sub-terrain
        # tiled 9x21 with a 20m border) to a single flat plane -- matches v1x1/asimov-1's own terrain setup
        # (same task/goal: walking, not rough-terrain generalization yet) and fixes envs spawning scattered
        # across a huge tiled grid (one curriculum "level"/row per difficulty) instead of a tight, easy-to-
        # view cluster near the origin.
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
        )
        self.scene.env_spacing = 2.5
        # terrain_levels_vel (base class default) indexes into the terrain generator's own multi-row grid --
        # meaningless/broken once terrain_type="plane" has no generator at all.
        self.curriculum.terrain_levels = None

        #self.events.randomize_imu_mount = EventTerm(
        #    func=randomize_imu_mount,
        #    mode="reset",
        #    params={
        #        "sensor_cfg": SceneEntityCfg("imu"),
        #        "pos_range": {
        #            "x": (-0.05, 0.05),
        #            "y": (-0.05, 0.05),
        #            "z": (-0.05, 0.05),
        #        },
        #        "rot_range": {
        #            "roll": (-0.1, 0.1),
        #            "pitch": (-0.1, 0.1),
        #            "yaw": (-0.1, 0.1),
        #        },
        #    },
        #)
        

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
        ]

        # rewards
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -5.0e-6
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_ang_vel_z_exp.weight = 5.0
        self.rewards.action_rate_l2.weight =-0.05     
        self.rewards.dof_acc_l2.weight =-1.25e-7      
        self.rewards.flat_orientation_l2.weight = -2

        # Walk
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4,0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-1, 1)


@configclass
class BdxrEnvCfg_PLAY(BdxrEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.1, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (0.00, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # remove random pushing event