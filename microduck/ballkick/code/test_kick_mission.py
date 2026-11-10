import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import kick_mission as km  # noqa: E402


def test_aligned_inside_the_lopsided_window():
    a = km.Aligner()
    assert a.pulse(0.0, 0.0, 0.0) is None
    assert a.pulse(-0.019, 0.011, math.radians(2.9)) is None   # ball close is fine
    assert a.pulse(0.011, -0.011, 0.0) is None


def test_heading_first_then_lateral_then_x():
    a = km.Aligner()
    assert a.pulse(0.03, 0.015, math.radians(5))[2] == "heading"
    assert a.pulse(0.03, 0.015, 0.0)[2] == "lateral"
    assert a.pulse(0.06, 0.0, 0.0)[2] == "step"


def test_no_step_from_close_range():
    """A 15-tick step's swing kicked the ball from 13-21 mm: back off instead."""
    a = km.Aligner()
    assert a.pulse(0.03, 0.0, 0.0)[2] == "backoff"
    assert a.pulse(0.046, 0.0, 0.0)[2] == "step"


def test_big_correction_with_ball_close_backs_off():
    a = km.Aligner()
    assert a.pulse(-0.01, 0.03, 0.0)[2] == "backoff"
    assert a.pulse(-0.01, 0.0, math.radians(20))[2] == "backoff"
    assert a.pulse(-0.01, 0.0, math.radians(10))[2] == "heading"  # small turns pivot in place
    assert a.pulse(0.01, 0.03, 0.0)[2] == "lateral"  # ball not close: strafe is fine


def test_backoff_budget_only_gates_the_big_correction_backoff():
    a = km.Aligner()
    # budget spent: correct in place instead of looping walk-ins
    assert a.pulse(-0.01, 0.03, 0.0, allow_backoff=False)[2] == "lateral"
    # ball 12-45 mm too far is still a backoff: can't step (swing hits it), can't kick
    assert a.pulse(0.03, 0.0, 0.0, allow_backoff=False)[2] == "backoff"


def test_stalled_axis_escalates_pulse_length():
    a = km.Aligner()
    k1 = a.pulse(0.0, 0.0, math.radians(4))[1]
    k2 = a.pulse(0.0, 0.0, math.radians(4))[1]   # no progress -> longer pulse
    assert k2 > k1
    assert a.pulse(0.0, 0.0, math.radians(1.5)) is None


def test_foot_relative_errors_absorb_a_staggered_kicking_foot():
    a = km.Aligner()
    home = np.array([0.0, -0.042])
    ball = a.target  # ball at the nominal trunk-frame offset, duck at origin facing +x
    ex, ey, _, _ = a.errors((0.0, 0.0, 0.0), ball, 0.0)
    assert abs(ex) < 1e-9 and abs(ey) < 1e-9
    ex, _, _, _ = a.errors((0.0, 0.0, 0.0), ball, 0.0, foot_local=home + (-0.02, 0.0), home_foot=home)
    assert abs(ex - 0.02) < 1e-9  # foot 20 mm back -> ball effectively 20 mm farther


def test_align_clamps():
    assert km.clamp_align(0.5, -0.4, 2.0) == (0.2, -0.15, 1.0)
