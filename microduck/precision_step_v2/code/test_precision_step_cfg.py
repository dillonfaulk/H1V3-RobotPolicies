import math

import torch

from mjlab_microduck.tasks import mdp as m
from mjlab_microduck.tasks import microduck_precision_step_env_cfg as ps


def test_actor_is_61d_layout_and_ball_blind():
    cfg = ps.make_microduck_precision_step_env_cfg()
    actor = cfg.observations["actor"].terms
    assert list(actor) == ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
                           "command", "head_command", "body_command"]
    assert not any("ball" in name for name in actor)
    assert actor["command"].params["command_name"] == "twist"
    assert isinstance(cfg.commands["twist"], m.StepTargetCommandCfg)
    assert "ball_position" in cfg.observations["critic"].terms


def test_trains_on_the_all_collisions_robot_with_a_ball():
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
    cfg = ps.make_microduck_precision_step_env_cfg()
    assert list(cfg.scene.entities)[0] == "robot"
    assert cfg.scene.entities["robot"] is MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
    assert "ball" in cfg.scene.entities
    assert cfg.sim.nconmax >= 200


def test_self_negating_penalties_have_positive_weights():
    """AGENTS sign rule: *_penalty functions return <= 0 and need weight > 0."""
    cfg = ps.make_microduck_precision_step_env_cfg()
    for name in ("foot_stagger", "overshoot", "ball_struck", "arrive_pose_l1"):
        term = cfg.rewards[name]
        assert term.func.__name__.endswith("_penalty"), name
        assert term.weight > 0, name
    for name in ("action_rate_l2", "body_ang_vel", "angular_momentum", "self_collisions"):
        assert cfg.rewards[name].weight < 0, name


def test_ball_strike_terminates_and_target_event_runs_after_ground_state():
    cfg = ps.make_microduck_precision_step_env_cfg()
    assert cfg.terminations["ball_struck"].time_out is False
    events = list(cfg.events)
    assert events.index("reset_step_target") > events.index("set_ground_state")


def test_offsets_respect_ranges_and_zero_bucket():
    torch.manual_seed(0)
    dx, dy, dyaw = m.sample_step_offsets(20000, "cpu", zero_prob=0.1, yaw_range=0.26)
    r = torch.sqrt(dx ** 2 + dy ** 2)
    assert r.max() <= 0.25 + 1e-6
    assert dyaw.abs().max() <= 0.26 + 1e-6
    zero = (r == 0) & (dyaw == 0)
    assert 0.07 < zero.float().mean() < 0.13
    assert (r < 0.08).float().mean() > 0.55          # mass at the small end
    assert (dx[r > 0] > 0).float().mean() > 0.6       # forward-biased (cone + half the all-around)


def test_kick_ball_sits_at_the_kick_offset_from_the_target_stance():
    goal = torch.tensor([[1.0, 2.0]])
    off = torch.tensor([[0.09, -0.042]])
    b = m.kick_ball_xy(goal, torch.tensor([0.0]), off)
    assert torch.allclose(b, torch.tensor([[1.09, 1.958]]))
    b = m.kick_ball_xy(goal, torch.tensor([math.pi / 2]), off)   # facing +y: forward is +y, right is +x
    assert torch.allclose(b, torch.tensor([[1.042, 2.09]]), atol=1e-6)


def test_world_to_yaw_frame_round_trip():
    bx, by = m.world_to_yaw_frame(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([math.pi / 2]))
    assert torch.allclose(bx, torch.tensor([1.0]), atol=1e-6) and torch.allclose(by, torch.tensor([0.0]), atol=1e-6)


def test_stagger_zero_at_home_and_counts_front_back_offset():
    yaw = torch.tensor([0.3])
    c, s = math.cos(0.3), math.sin(0.3)
    def w(x, y):  # yaw-frame point -> world
        return torch.tensor([[c * x - s * y, s * x + c * y]])
    home = m.foot_stagger_from_sites(w(0.0, 0.042), w(0.0, -0.042), yaw, 0.084)
    assert home.item() < 1e-6
    staggered = m.foot_stagger_from_sites(w(0.0, 0.042), w(-0.02, -0.042), yaw, 0.084)
    assert abs(staggered.item() - 0.02) < 1e-6
    wide = m.foot_stagger_from_sites(w(0.0, 0.052), w(0.0, -0.042), yaw, 0.084)
    assert abs(wide.item() - 0.005) < 1e-6   # width error counts half


def test_kick_spot_is_clear_of_the_footprint_but_the_feet_are_not():
    pad = ps.BALL_RADIUS + 0.006
    def clear(x, y, yaw=0.0):
        return bool(m.ball_clear_of_footprint(torch.tensor([x]), torch.tensor([y]), torch.tensor([yaw]),
                                              ps.FOOTPRINT_X, ps.FOOTPRINT_Y, pad))
    assert clear(0.09, -0.042)                 # nominal kick spot from a zero-offset target
    assert clear(0.076, -0.057)                # near BallKick's closest noisy spawn (75 mm)
    assert not clear(0.05, -0.042)             # would overlap the right toe
    assert not clear(0.0, 0.0)
    # same geometry when the duck faces +y
    assert clear(0.042, 0.09, math.pi / 2) and not clear(0.042, 0.05, math.pi / 2)


# ── PrecisionStep v2: ball on both sides (fork-local, 2026-11-13) ────────────

def test_ball_sides_default_is_right_only(monkeypatch):
    import importlib
    monkeypatch.delenv("MICRODUCK_PSTEP_BALL_SIDES", raising=False)
    mod = importlib.reload(ps)
    assert mod.BALL_LEFT_PROB == 0.0
    cfg = mod.make_microduck_precision_step_env_cfg()
    assert cfg.events["reset_step_target"].params["ball_left_prob"] == 0.0
    assert mod.MicroduckPrecisionStepRlCfg.experiment_name == "precision_step"


def test_ball_sides_both_env_sets_half_left_and_v2_experiment(monkeypatch):
    import importlib
    monkeypatch.setenv("MICRODUCK_PSTEP_BALL_SIDES", "both")
    try:
        mod = importlib.reload(ps)
        cfg = mod.make_microduck_precision_step_env_cfg()
        assert cfg.events["reset_step_target"].params["ball_left_prob"] == 0.5
        assert mod.MicroduckPrecisionStepRlCfg.experiment_name == "precision_step_v2"
    finally:
        monkeypatch.delenv("MICRODUCK_PSTEP_BALL_SIDES")
        importlib.reload(ps)


def _stub_env(n):
    from types import SimpleNamespace as NS
    qpos = torch.zeros(n, 7)
    qpos[:, 2] = 0.115
    qpos[:, 3] = 1.0                                   # identity quat (wxyz), facing +x
    written = {}
    robot = NS(indexing=NS(free_joint_q_adr=torch.arange(7)))
    ball = NS(write_root_link_pose_to_sim=lambda p, ids: written.__setitem__("pose", p),
              write_root_link_velocity_to_sim=lambda v, ids: None)
    env = NS(device="cpu", num_envs=n, sim=NS(data=NS(qpos=qpos)))
    env.scene = type("S", (), {"__getitem__": lambda self, k: {"robot": robot, "ball": ball}[k],
                               "terrain": NS(env_origins=torch.zeros(n, 3))})()
    return env, written


def _ball_side_counts(left_prob, n=4000):
    torch.manual_seed(0)
    env, written = _stub_env(n)
    m.reset_step_target(env, torch.arange(n), ball_prob=1.0, ball_left_prob=left_prob)
    st = env._pstep
    goal, bxy, has = st["goal"], st["ball_home"], st["has_ball"]
    # ball position in the TARGET stance's yaw frame
    lx, ly = m.world_to_yaw_frame(bxy[:, 0] - goal[:, 0], bxy[:, 1] - goal[:, 1], goal[:, 2])
    lx, ly = lx[has], ly[has]
    assert ((lx > 0.09 - 0.0151) & (lx < 0.09 + 0.0151)).all()
    assert ((ly.abs() > 0.042 - 0.0151) & (ly.abs() < 0.042 + 0.0151)).all()
    return int((ly > 0).sum()), int((ly < 0).sum())


def test_v1_reset_puts_every_ball_on_the_right():
    left, right = _ball_side_counts(0.0)
    assert left == 0 and right > 3000


def test_v2_reset_mirrors_about_half_the_balls_to_the_left_kick_spot():
    left, right = _ball_side_counts(0.5)
    assert 0.45 < left / (left + right) < 0.55
