import importlib

from mjlab_microduck.tasks import microduck_ball_kick_env_cfg as bk


def test_target_speed_default_is_upstream(monkeypatch):
    monkeypatch.delenv("MICRODUCK_BALL_TARGET_SPEED", raising=False)
    mod = importlib.reload(bk)
    assert mod.BALL_TARGET_SPEED == 1.0
    cfg = mod.make_microduck_ball_kick_env_cfg()
    assert cfg.rewards["ball_forward_velocity"].params["max_speed"] == 1.0
    assert cfg.rewards["ball_speed_overshoot"].params["target_speed"] == 1.0
    assert cfg.rewards["ball_forward_velocity"].weight == 12.0
    assert cfg.rewards["ball_speed_overshoot"].weight == -4.0


def test_target_speed_env_override_reaches_both_kick_terms(monkeypatch):
    monkeypatch.setenv("MICRODUCK_BALL_TARGET_SPEED", "0.25")
    try:
        mod = importlib.reload(bk)
        cfg = mod.make_microduck_ball_kick_env_cfg()
        assert cfg.rewards["ball_forward_velocity"].params["max_speed"] == 0.25
        assert cfg.rewards["ball_speed_overshoot"].params["target_speed"] == 0.25
    finally:
        monkeypatch.delenv("MICRODUCK_BALL_TARGET_SPEED")
        importlib.reload(bk)


def test_actor_stays_ball_blind_and_61d():
    cfg = bk.make_microduck_ball_kick_env_cfg()
    actor = cfg.observations["actor"].terms
    assert not any("ball" in name for name in actor)
    assert list(actor) == ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
                           "command", "head_command", "body_command"]
    assert "ball_position" in cfg.observations["critic"].terms


def test_kick_foot_default_is_right(monkeypatch):
    monkeypatch.delenv("MICRODUCK_KICK_FOOT", raising=False)
    mod = importlib.reload(bk)
    assert mod.KICK_FOOT == "right"
    cfg = mod.make_microduck_ball_kick_env_cfg()
    assert cfg.events["reset_ball"].params["offset"] == (0.09, -0.042)


def test_kick_foot_env_left_mirrors_ball_support_and_experiment(monkeypatch):
    monkeypatch.setenv("MICRODUCK_KICK_FOOT", "left")
    try:
        mod = importlib.reload(bk)
        assert mod.KICK_FOOT == "left"
        cfg = mod.make_microduck_ball_kick_env_cfg()
        assert cfg.events["reset_ball"].params["offset"] == (0.09, 0.042)   # +y = left toe
        support = [s for s in cfg.scene.sensors if s.name == "support_foot_ground_contact"][0]
        assert "right_foot_collision" in support.primary.pattern               # right leg supports
        assert mod.MicroduckBallKickRlCfg.experiment_name == "ball_kick_left"
    finally:
        monkeypatch.delenv("MICRODUCK_KICK_FOOT")
        importlib.reload(bk)
