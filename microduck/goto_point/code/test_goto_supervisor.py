import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import goto_supervisor as gs  # noqa: E402

ARENA = (-2.5, 2.5, -2.5, 2.5)
UP = dict(tilt_deg=3.0, trunk_z=0.115)


def make(wps, **kw):
    events = []
    sup = gs.GotoSupervisor(wps, ARENA, gs.GotoConfig(**kw), log=events.append)
    return sup, events


def tick(sup, t, pose, tilt_deg=3.0, trunk_z=0.115):
    sup.update(t, pose, tilt_deg, trunk_z)
    return sup.command(t)


def names(events):
    return [e["event"] for e in events if e["event"] != "GO_TO_TRACE"]


def test_clamps_forward_only_no_strafe():
    assert gs.clamp_twist(0.6, 0.3, 2.0) == (gs.VX_MAX, 0.0, gs.WZ_ABS_MAX)
    assert gs.clamp_twist(-0.2, -0.3, -2.0) == (0.0, 0.0, -gs.WZ_ABS_MAX)


def test_goal_in_base_frame():
    x, y = gs.goal_in_base((1.0, 1.0, math.pi / 2), (1.0, 2.0))  # goal straight ahead when facing +y
    assert abs(x - 1.0) < 1e-9 and abs(y) < 1e-9


def test_never_commands_a_creep():
    """vx is 0 or cruise: the policy's walk response is a step, a taper would stall outside the radius."""
    sup, _ = make([(1.0, 0.0)])
    seen = set()
    for k in range(200):
        x = k * 0.005  # walk the robot straight at the goal, all the way in
        vx, vy, wz = tick(sup, k * 0.02, (x, 0.0, 0.0))
        seen.add(vx)
        assert vy == 0.0
    assert seen <= {0.0, sup.cfg.cruise_vx}
    assert sup.arrived == [1] and sup.state == "DONE"


def test_arrival_pops_queue_and_finishes():
    sup, ev = make([(0.5, 0.0), (0.5, 0.5)])
    tick(sup, 0.0, (0.0, 0.0, 0.0))
    assert tick(sup, 0.02, (0.45, 0.0, 0.0)) == (0.0, 0.0, 0.0)
    assert sup.arrived == [1] and sup.current[0] == 2
    tick(sup, 0.04, (0.5, 0.45, math.pi / 2))
    assert sup.state == "DONE"
    assert names(ev) == ["GO_TO_START", "GO_TO_ARRIVE", "GO_TO_START", "GO_TO_ARRIVE", "GO_TO_DONE"]


def test_geofence_rejects_before_motion():
    sup, ev = make([(1.0, 0.0), (9.0, 0.0), (1.0, 1.0)])
    assert names(ev) == ["GO_TO_REJECT"] and ev[0]["wp"] == 2
    assert [i for i, _ in sup.queue] == [1, 3]
    sup2, ev2 = make([(9.0, 9.0)])
    assert sup2.state == "DONE" and tick(sup2, 0.0, (0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)


def test_goal_behind_pivots_with_floor_then_arcs():
    sup, _ = make([(-1.0, 0.1)])
    vx, _, wz = tick(sup, 0.0, (0.0, 0.0, 0.0))
    assert vx == 0.0 and abs(wz) == sup.cfg.pivot_wz >= 0.5  # in-place turns are dead <= 0.4 rad/s
    # still pivoting at 40 deg (hysteresis), arcing below the 20 deg exit
    assert tick(sup, 0.02, (0.0, 0.0, math.pi - math.radians(40)))[0] == 0.0
    vx, _, wz = tick(sup, 0.04, (0.0, 0.0, math.pi - math.radians(10)))
    assert vx == sup.cfg.cruise_vx and abs(wz) < sup.cfg.pivot_wz


def test_close_side_goal_pivots_instead_of_orbiting():
    sup, _ = make([(0.2, 0.2)])  # 45 deg off, inside the 0.2 m min arc radius circle
    vx, _, wz = tick(sup, 0.0, (0.0, 0.0, 0.0))
    assert vx == 0.0 and wz > 0


def test_fall_commands_zero_until_upright_and_settled():
    sup, ev = make([(2.0, 0.0)])
    tick(sup, 0.0, (0.0, 0.0, 0.0))
    assert tick(sup, 0.02, (0.3, 0.0, 0.0), tilt_deg=80.0, trunk_z=0.05) == (0.0, 0.0, 0.0)
    assert sup.state == "RECOVERING" and sup.falls == 1
    t = 0.04
    while t < 0.5:  # upright but not settled yet
        assert tick(sup, t, (0.3, 0.0, 0.0)) == (0.0, 0.0, 0.0)
        t += 0.02
    while t < 1.2:
        cmd = tick(sup, t, (0.3, 0.0, 0.0))
        t += 0.02
    assert sup.state == "NAVIGATE" and cmd[0] == sup.cfg.cruise_vx
    assert names(ev) == ["GO_TO_START", "FALL", "RECOVERED"]
    assert ev[1]["wp"] == 1


def test_watchdog_zeroes_on_stale_pose():
    sup, ev = make([(2.0, 0.0)])
    assert tick(sup, 0.0, (0.0, 0.0, 0.0))[0] > 0
    assert sup.command(0.25) == (0.0, 0.0, 0.0)  # no pose for 250 ms
    assert "WATCHDOG" in names(ev)
    assert tick(sup, 0.27, (0.0, 0.0, 0.0))[0] > 0  # fresh pose -> resumes


def test_estop_holds_zero():
    sup, _ = make([(2.0, 0.0)])
    sup.estop = True
    assert tick(sup, 0.0, (0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
    sup.estop = False
    assert tick(sup, 0.02, (0.0, 0.0, 0.0))[0] > 0
