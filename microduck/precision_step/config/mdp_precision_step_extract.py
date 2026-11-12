"""PrecisionStep section of microduck_rl src/mjlab_microduck/tasks/mdp.py (extract, commit a74d061d).

Not importable on its own: relies on mdp.py helpers (pose_target_match, pose_l1_penalty, ...) and mjlab."""

# =============================================================================
# PrecisionStep task — step to a small target pose and stop dead, feet square
# =============================================================================
#
# Command contract (Dillon's call, 2026-11-11, "option A"): the 3 twist slots of
# the unified 61D layout carry the REMAINING offset to the target pose, in the
# trunk's current yaw frame: [dx (m, forward), dy (m, left), dyaw (rad)].
# Head/body slots stay zero-padded. At deployment the supervisor recomputes the
# same three numbers every tick from the pose estimate. VelStand reads those
# slots as a velocity; this policy reads them as an offset. The supervisor picks
# the ONNX, so it fills the slots to match (precedent: the posture flag lives in
# the twist vx slot).
#
# Per-episode state lives on the env (lazily allocated), set by the
# ``reset_step_target`` event (events run BEFORE the command manager resets).


def _pstep_state(env: ManagerBasedRlEnv) -> dict:
    if not hasattr(env, "_pstep"):
        n, d = env.num_envs, env.device
        env._pstep = {
            "goal": torch.zeros(n, 3, device=d),        # world x, y, yaw
            "dir0": torch.zeros(n, 2, device=d),        # unit start->goal (0 if no translation)
            "prev_dist": torch.zeros(n, device=d),      # for potential-based progress
            "has_ball": torch.zeros(n, dtype=torch.bool, device=d),
            "ball_home": torch.zeros(n, 2, device=d),   # ball xy at reset
        }
    return env._pstep


def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def world_to_yaw_frame(dx: torch.Tensor, dy: torch.Tensor, yaw: torch.Tensor):
    """Rotate a world-frame XY delta into a frame with heading ``yaw``."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def sample_step_offsets(
    n: int,
    device,
    near_r: tuple = (0.0, 0.08),
    far_r: tuple = (0.08, 0.25),
    far_prob: float = 0.35,
    fwd_cone_prob: float = 0.5,
    fwd_cone_half: float = math.pi / 4,
    yaw_range: float = 0.26,
    zero_prob: float = 0.1,
):
    """(dx, dy, dyaw) target offsets in the START yaw frame. Pure, testable.

    Most mass at small radii (where the pulse aligner fails today), a far tail up
    to the 0.25 m final-approach zone, half the bearings in a forward cone (the
    mission approaches from behind the ball), the rest all-around (symmetry: any
    direction, either foot). An explicit exact-zero bucket trains "hold still".
    """
    far = torch.rand(n, device=device) < far_prob
    u = torch.rand(n, device=device)
    r = torch.where(far, far_r[0] + u * (far_r[1] - far_r[0]), near_r[0] + u * (near_r[1] - near_r[0]))
    cone = torch.rand(n, device=device) < fwd_cone_prob
    b_cone = (torch.rand(n, device=device) * 2 - 1) * fwd_cone_half
    b_all = (torch.rand(n, device=device) * 2 - 1) * math.pi
    bearing = torch.where(cone, b_cone, b_all)
    dyaw = (torch.rand(n, device=device) * 2 - 1) * yaw_range
    zero = torch.rand(n, device=device) < zero_prob
    r = torch.where(zero, torch.zeros_like(r), r)
    dyaw = torch.where(zero, torch.zeros_like(dyaw), dyaw)
    return r * torch.cos(bearing), r * torch.sin(bearing), dyaw


def kick_ball_xy(goal_xy: torch.Tensor, goal_yaw: torch.Tensor, offset_xy: torch.Tensor) -> torch.Tensor:
    """Ball center for a kick stance at (goal_xy, goal_yaw): offset in that yaw frame."""
    c, s = torch.cos(goal_yaw), torch.sin(goal_yaw)
    return torch.stack(
        [goal_xy[:, 0] + c * offset_xy[:, 0] - s * offset_xy[:, 1],
         goal_xy[:, 1] + s * offset_xy[:, 0] + c * offset_xy[:, 1]], dim=1)


def ball_clear_of_footprint(dx_w, dy_w, yaw, keepout_x, keepout_y, pad):
    """True where a ball center at world delta (dx_w, dy_w) from the trunk lies
    outside the footprint box (trunk yaw frame) expanded by ``pad``."""
    bx, by = world_to_yaw_frame(dx_w, dy_w, yaw)
    inside = (bx > keepout_x[0] - pad) & (bx < keepout_x[1] + pad) & \
             (by > keepout_y[0] - pad) & (by < keepout_y[1] + pad)
    return ~inside


def reset_step_target(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    near_r: tuple = (0.0, 0.08),
    far_r: tuple = (0.08, 0.25),
    far_prob: float = 0.35,
    fwd_cone_prob: float = 0.5,
    yaw_range: float = 0.26,
    zero_prob: float = 0.1,
    ball_prob: float = 0.7,
    ball_offset: tuple = (0.09, -0.042),
    ball_noise_xy: float = 0.015,
    ball_radius: float = 0.035,
    keepout_x: tuple = (-0.033, 0.034),
    keepout_y: tuple = (-0.079, 0.079),
    keepout_margin: float = 0.006,
    ball_park_dist: float = 3.0,
    asset_name: str = "ball",
):
    """Sample the episode's target pose and place (or park) the ball.

    The ball goes where it would sit if this were a real final approach: at the
    BallKick placement (0.09, -0.042 with the same +-15 mm noise) relative to the
    TARGET stance. The actor never sees it; touching it ends the episode. Ball
    episodes whose ball would overlap the duck's HOME footprint at the START
    pose (``keepout_x/y``: collision-geom box within one ball diameter of the
    soles, measured 2026-11-11, start yaw frame; expanded by the ball radius +
    ``keepout_margin``) are made ball-free instead; parked balls sit
    ``ball_park_dist`` away. Must run AFTER set_ground_state (reads the
    final robot pose from qpos; root_link_pos_w lags until the next forward()).
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device)
    st = _pstep_state(env)
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]
    n = len(env_ids)

    root = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
    x0, y0 = root[:, 0], root[:, 1]
    yaw0 = _yaw_from_quat_wxyz(root[:, 3:7])

    dx, dy, dyaw = sample_step_offsets(
        n, env.device, near_r, far_r, far_prob, fwd_cone_prob, math.pi / 4, yaw_range, zero_prob)
    c, s = torch.cos(yaw0), torch.sin(yaw0)
    gx = x0 + c * dx - s * dy
    gy = y0 + s * dx + c * dy
    gyaw = wrap_to_pi(yaw0 + dyaw)
    st["goal"][env_ids] = torch.stack([gx, gy, gyaw], dim=1)
    dist = torch.sqrt(dx * dx + dy * dy)
    st["prev_dist"][env_ids] = dist
    dir_w = torch.stack([gx - x0, gy - y0], dim=1)
    st["dir0"][env_ids] = torch.where(
        dist.unsqueeze(1) > 1e-6, dir_w / dist.clamp_min(1e-6).unsqueeze(1), torch.zeros_like(dir_w))

    off = torch.tensor(ball_offset, device=env.device, dtype=torch.float).repeat(n, 1)
    off += (torch.rand(n, 2, device=env.device) * 2.0 - 1.0) * ball_noise_xy
    bxy = kick_ball_xy(torch.stack([gx, gy], dim=1), gyaw, off)
    clear = ball_clear_of_footprint(bxy[:, 0] - x0, bxy[:, 1] - y0, yaw0, keepout_x, keepout_y,
                                    ball_radius + keepout_margin)
    has = (torch.rand(n, device=env.device) < ball_prob) & clear
    park = torch.stack([x0 + ball_park_dist, y0], dim=1)
    bxy = torch.where(has.unsqueeze(1), bxy, park)
    st["has_ball"][env_ids] = has
    st["ball_home"][env_ids] = bxy

    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0:2] = bxy
    pose[:, 2] = env.scene.terrain.env_origins[env_ids, 2] + ball_radius
    pose[:, 3] = 1.0
    ball.write_root_link_pose_to_sim(pose, env_ids)
    ball.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids)


class StepTargetCommand(CommandTerm):
    """Remaining offset to the episode's target pose, in the trunk's yaw frame.

    command = [dx, dy, dyaw]. The goal itself is sampled by ``reset_step_target``
    (an event, which runs before the command manager resets). Metrics hold the
    LATEST error (not an accumulation), so the value logged at reset is the
    terminal error of the finished episode.
    """

    cfg: "StepTargetCommandCfg"

    def __init__(self, cfg: "StepTargetCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene["robot"]
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["final_pos_err"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["final_yaw_err"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _offset(self, x, y, yaw, env_ids=None):
        goal = _pstep_state(self._env)["goal"]
        if env_ids is not None:
            goal = goal[env_ids]
        bx, by = world_to_yaw_frame(goal[:, 0] - x, goal[:, 1] - y, yaw)
        return torch.stack([bx, by, wrap_to_pi(goal[:, 2] - yaw)], dim=1)

    def _update_metrics(self) -> None:
        self.metrics["final_pos_err"][:] = torch.linalg.norm(self._command[:, :2], dim=1)
        self.metrics["final_yaw_err"][:] = self._command[:, 2].abs()

    def _update_command(self) -> None:
        pos = self.robot.data.root_link_pos_w
        yaw = _yaw_from_quat_wxyz(self.robot.data.root_link_quat_w)
        self._command[:] = torch.nan_to_num(self._offset(pos[:, 0], pos[:, 1], yaw), nan=0.0)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        root = self._env.sim.data.qpos[env_ids][:, self.robot.indexing.free_joint_q_adr]
        yaw = _yaw_from_quat_wxyz(root[:, 3:7])
        self._command[env_ids] = self._offset(root[:, 0], root[:, 1], yaw, env_ids)


@dataclass(kw_only=True)
class StepTargetCommandCfg(CommandTermCfg):
    def build(self, env: ManagerBasedRlEnv) -> "StepTargetCommand":
        return StepTargetCommand(self, env)


def step_target_position(env: ManagerBasedRlEnv, std: float, command_name: str = "twist") -> torch.Tensor:
    """Gaussian on the remaining XY distance to the target (>= 0)."""
    cmd = env.command_manager.get_command(command_name)
    return torch.exp(-(torch.linalg.norm(cmd[:, :2], dim=1) / std) ** 2)


def step_target_heading(env: ManagerBasedRlEnv, std: float, command_name: str = "twist") -> torch.Tensor:
    """Gaussian on the remaining heading error (>= 0)."""
    cmd = env.command_manager.get_command(command_name)
    return torch.exp(-(cmd[:, 2] / std) ** 2)


def step_target_progress(
    env: ManagerBasedRlEnv, max_speed: float = 0.3, command_name: str = "twist"
) -> torch.Tensor:
    """Potential-based progress: (previous distance - distance) / dt, clamped.

    Pays approach, charges retreat, pays zero for holding: unfarmable, and the
    only gradient for far targets where the position Gaussians are ~flat.
    """
    st = _pstep_state(env)
    d = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    rate = (st["prev_dist"] - d) / env.step_dt
    st["prev_dist"][:] = d
    return torch.nan_to_num(rate, nan=0.0).clamp(-max_speed, max_speed)


def step_target_settled(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    pos_tol: float = 0.015,
    yaw_tol: float = 0.05,
    lin_std: float = 0.03,
    ang_std: float = 0.2,
    command_name: str = "twist",
) -> torch.Tensor:
    """Arrive and STOP DEAD: inside tolerance, both feet down, trunk still (>= 0).

    Gated on a GOOD state (on target), never a bad one.
    """
    robot: Entity = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    on = (torch.linalg.norm(cmd[:, :2], dim=1) < pos_tol) & (cmd[:, 2].abs() < yaw_tol)
    found = env.scene.sensors[sensor_name].data.found
    both_down = (found > 0).all(dim=-1) if found.dim() > 1 else found > 0
    v = torch.linalg.norm(robot.data.root_link_lin_vel_b[:, :2], dim=1)
    w = robot.data.root_link_ang_vel_b[:, 2].abs()
    still = torch.exp(-(v / lin_std) ** 2 - (w / ang_std) ** 2)
    return torch.nan_to_num(on.float() * both_down.float() * still, nan=0.0)


def foot_stagger_from_sites(
    left_xy: torch.Tensor, right_xy: torch.Tensor, yaw: torch.Tensor, home_width: float, width_weight: float = 0.5
) -> torch.Tensor:
    """|front-back offset| + width_weight * |width - home_width| of the feet, in the trunk yaw frame (>= 0)."""
    fx, fy = world_to_yaw_frame(left_xy[:, 0] - right_xy[:, 0], left_xy[:, 1] - right_xy[:, 1], yaw)
    return fx.abs() + width_weight * (fy - home_width).abs()


def step_foot_stagger_penalty(
    env: ManagerBasedRlEnv,
    near: float = 0.04,
    home_width: float = 0.084,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    command_name: str = "twist",
) -> torch.Tensor:
    """SELF-NEGATING (<= 0, positive weight): feet not square, only near the target.

    Attacks the #1 BallKick miss mode (support foot behind home) at the root: the
    stance at arrival must already be the kick's HOME stance.
    """
    robot: Entity = env.scene[asset_cfg.name]
    sites = robot.data.site_pos_w[:, asset_cfg.site_ids, :2]
    yaw = _yaw_from_quat_wxyz(robot.data.root_link_quat_w)
    st = foot_stagger_from_sites(sites[:, 0], sites[:, 1], yaw, home_width)
    near_m = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < near
    return -torch.nan_to_num(st, nan=0.0).clamp(max=0.1) * near_m.float()


def step_overshoot_penalty(
    env: ManagerBasedRlEnv,
    margin: float = 0.01,
    cap: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """SELF-NEGATING (<= 0): distance past the goal along the start->goal line, beyond ``margin``."""
    st = _pstep_state(env)
    robot: Entity = env.scene[asset_cfg.name]
    rel = robot.data.root_link_pos_w[:, :2] - st["goal"][:, :2]
    past = (rel * st["dir0"]).sum(dim=1) - margin
    return -torch.nan_to_num(past, nan=0.0).clamp(0.0, cap)


def _ball_displaced(env: ManagerBasedRlEnv, threshold: float, asset_name: str) -> torch.Tensor:
    st = _pstep_state(env)
    ball: Entity = env.scene[asset_name]
    moved = torch.linalg.norm(ball.data.root_link_pos_w[:, :2] - st["ball_home"], dim=1)
    return st["has_ball"] & (torch.nan_to_num(moved, nan=1.0) > threshold)


def ball_displaced_termination(
    env: ManagerBasedRlEnv, threshold: float = 0.01, asset_name: str = "ball"
) -> torch.Tensor:
    """Episode fails as soon as the robot moves the ball more than ``threshold``."""
    return _ball_displaced(env, threshold, asset_name)


def ball_displaced_penalty(
    env: ManagerBasedRlEnv, threshold: float = 0.01, asset_name: str = "ball"
) -> torch.Tensor:
    """SELF-NEGATING (<= 0, positive weight): -1 on the step the ball is displaced (episode ends there)."""
    return -_ball_displaced(env, threshold, asset_name).float()


def _near_target(env: ManagerBasedRlEnv, near: float, command_name: str) -> torch.Tensor:
    return (torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < near).float()


def step_terminal_pose_match(
    env: ManagerBasedRlEnv,
    std: float,
    joint_indices: list,
    near: float = 0.03,
    command_name: str = "twist",
) -> torch.Tensor:
    """Gaussian leg pose vs HOME, only near the target (>= 0).

    model_1000 of the first run reached every target in a deep crouch (hips ~30
    deg, knees ~36 deg off HOME) and the kick, trained from HOME +-0.05 rad,
    never swung from it (0/60 missions). The arrival stance must BE the kick's
    start stance: tight std, paid only where the stepping is over.
    """
    return pose_target_match(env, std=std, joint_indices=joint_indices) * _near_target(env, near, command_name)


def step_terminal_pose_penalty(
    env: ManagerBasedRlEnv,
    joint_indices: list,
    near: float = 0.03,
    command_name: str = "twist",
) -> torch.Tensor:
    """SELF-NEGATING (<= 0, positive weight): mean |leg joint - HOME|, only near the target.

    The L1 companion gives a constant gradient out of the crouch, where the
    tight Gaussian above is flat.
    """
    return pose_l1_penalty(env, joint_indices=joint_indices) * _near_target(env, near, command_name)
