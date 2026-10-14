from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg # noqa: F401
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

# Dynamically get the directory where this bdxr.py script is located
BDXR_ASSETS_DIR = Path(__file__).resolve().parent


def _spawn_bdxr_with_full_contact_sensors(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Wraps the stock URDF spawn function to fix a contact-sensor activation gap.

    isaaclab.sim.schemas.activate_contact_sensors() (triggered by UrdfFileCfg's own
    activate_contact_sensors=True) walks the spawned prim tree but stops descending into a
    branch the moment it finds a RigidBodyAPI prim -- it assumes rigid bodies are never nested
    inside one another. This URDF's on-the-fly conversion places each leg link as a USD child
    of its parent link (mirroring the kinematic chain), so only the outermost rigid body
    ("base_link") actually gets the PhysX contact-report API; all 10 leg/foot bodies nested
    below it are silently skipped (confirmed via direct stage probe). bipedal_air_time_reward,
    foot_clearance_reward, and foot_slip_penalty all depend on foot contact data, so this isn't
    optional -- re-walk the tree ourselves (without the stop-at-first-match assumption) and
    activate every rigid body found, not just the first one per branch.
    """
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_urdf
    from isaaclab.sim.schemas import activate_contact_sensors
    import omni.usd
    from pxr import UsdPhysics

    prim = spawn_from_urdf(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)

    if cfg.activate_contact_sensors:
        stage = omni.usd.get_context().get_stage()
        root_prim = stage.GetPrimAtPath(prim_path)

        def walk(p):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                activate_contact_sensors(p.GetPath().pathString, threshold=0.0, stage=stage)
            for child in p.GetChildren():
                walk(child)

        walk(root_prim)

    return prim


BDX_R_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        func=_spawn_bdxr_with_full_contact_sensors,
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        # Now it properly points to the URDF.urdf in the exact same directory
        asset_path=str(Path(__file__).resolve().parents[2] / "data/Robots/BDXR/URDF.urdf"),
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.33),
    ),
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Hip_Yaw", ".*_Hip_Roll", ".*_Hip_Pitch", ".*_Knee", ".*_Ankle"],
            stiffness={
                ".*_Hip_Yaw": 78.957,
                ".*_Hip_Roll": 78.957,
                ".*_Hip_Pitch": 78.957,
                ".*_Knee": 78.957,
                ".*_Ankle": 16.581,
            },
            damping={
                ".*_Hip_Yaw": 5.027,
                ".*_Hip_Roll": 5.027,
                ".*_Hip_Pitch": 5.027,
                ".*_Knee": 5.027,
                ".*_Ankle": 1.056,
            },
            armature={
                ".*_Hip_Yaw": 0.02,
                ".*_Hip_Roll": 0.02,
                ".*_Hip_Pitch": 0.02,
                ".*_Knee": 0.02,
                ".*_Ankle": 0.0042,
            },
            effort_limit_sim={
                ".*_Hip_Yaw": 42.0,
                ".*_Hip_Roll": 42.0,
                ".*_Hip_Pitch": 42.0,
                ".*_Knee": 42.0,
                ".*_Ankle": 11.9,
            },
            velocity_limit_sim={
                ".*_Hip_Yaw": 18.849,
                ".*_Hip_Roll": 18.849,
                ".*_Hip_Pitch": 18.849,
                ".*_Knee": 18.849,
                ".*_Ankle": 37.699,
            },
            min_delay=0,
            max_delay=0
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""Configuration for the Disney BD-X robot with implicit actuator model."""