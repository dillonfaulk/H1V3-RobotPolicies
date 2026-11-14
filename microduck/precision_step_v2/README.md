# Microduck PrecisionStep v2 (ball on both sides)

The final-approach policy from [PrecisionStep v1](../precision_step/README.md), retrained from scratch with the ball on either side. v1 only ever saw the ball on the right, where the right foot kicks from. That was enough for the left foot at its normal stance, but when the left-foot stance moved 9 mm closer, v1's stepping foot clipped the ball. v2 fixes that, and it's a drop-in replacement: same contract, same mission, both feet. **Sim only**; nothing here has run on the physical robot.

## Results

### The regression test v2 exists for

Left-foot missions with the stance moved +9 mm toward the ball. This is the geometry where v1 bumped the ball.

| Left foot, +9 mm stance, 2 × 30 headless (seeds 0, 1) | v1 (iteration 3999) | **v2 (iteration 2500)** |
|---|---|---|
| Kicked | 58/60 | **60/60** |
| Bumped balls | 2 | **0** |
| Falls | 0 | 0 |
| Dillon's live viewer, 30 back to back | — | **30/30**, 0 bumps, 0 falls |

At +9 mm the left kicks land about 5° right of the line: −4.8° median headless, −4.5° in the viewer. That's expected. The left foot's +11.2° aim was calibrated at 0 mm, and moving the stance changes the drift. Re-aiming happens in BallKick v2's calibration step.

### Missions with both kicks

The current kick policies (right iteration 1000, left iteration 1250) run with their published calibrations, through v2.

| 2 × 30 headless | Right foot (+10 mm, aim −17.9°) | Left foot (0 mm, aim +11.2°) |
|---|---|---|
| Kicked / bumps / falls | **60/60** / 0 / 0 | **60/60** / 0 / 0 |
| Kick direction, + = left (median, range) | +0.6° (−4.3 to +5.6) | +0.9° (−4.8 to +4.5) |
| Through v1 (reference) | +0.1° (−3.8 to +4.0) | −0.2° (−3.5 to +3.6) |
| Ball roll, median | 1.21 m (v1: 1.20) | 1.17 m (v1: 1.21) |
| Worst leg joint vs HOME at the kick swap (median, worst) | 3.7°, 6.4° | 3.6°, 5.4° |

### Placement battery, by ball side

200 episodes, swapped in after 1.5 s of VelStand standing, seed 1. v1's published battery was 50 episodes at seed 1; the side-split script reproduces it exactly on the right side.

| | Ball right: kick-ready | Ball left: kick-ready | Position error, median (R / L) | Struck (R / L) | Falls |
|---|---|---|---|---|---|
| v1, iteration 3999 | 106/112 | 86/88 | 4.8 / 4.0 mm | 5/97 / 1/73 | 4 |
| **v2, iteration 2500** | 103/112 | 80/88 | **3.2 / 3.0 mm** | 6/97 / 1/73 | 2 |

**On this battery v2 is slightly behind v1, and I'm reporting that rather than hiding it.**
- **Right side:** it's 3 episodes short, within noise. v1 lost 3 of its 6 there to falls, against v2's 1.
- **Left side:** all 6 extra failures are the posture rule. KICK-READY requires every leg joint within 0.10 rad (5.7°) of HOME, and v2 sometimes stops on target a degree or two past that.

The missions show that rule is stricter than the kicks need: every mission kicked, including ones with legs up to 8° off HOME.

## The policy

| | |
|---|---|
| File | `precision_step_v2_model_2500.onnx` |
| Task | `Mjlab-PrecisionStep-Flat-MicroDuck` with `MICRODUCK_PSTEP_BALL_SIDES=both` (fork-local env var; unset = v1) |
| Contract | `[1, 61]` → `[1, 14]`, 50 Hz, normalizer baked in. Slots 48–50 = remaining offset to the target pose `[dx, dy, dyaw]`, the same as v1. |
| ONNX check | max abs diff vs the PyTorch actor 1.2e-6 over 1,000 random inputs |

## What changed from v1 (one variable)

- **Ball side.** In ball episodes (70%), the ball's sideways offset flips to the left foot's kick spot half the time: (0.09, +0.042) instead of (0.09, −0.042) from the target stance, with the same ±15 mm noise and the same 1 cm "struck" rule. The actor still never sees the ball. This is `ball_left_prob=0.5` in `reset_step_target` (see `config/mdp_precision_step_extract.py`).
- **Training history, not recipe:** v1 got its arrival-posture rewards partway through, at iteration 1250, after a crouch failure. v2 has them from iteration 0. The final reward set is the same.
- **Training:** from scratch, 2,048 envs, 4,000 iterations on GPU0 (11.8 GB, 1 h 38 min). Every penalty term logged ≤ 0 on every readout.

## Why iteration 2500

| Checkpoint | 2000 | **2500** | 3000 | 3500 | 3999 |
|---|---|---|---|---|---|
| Battery kick-ready, right / left | 98 / 86 | **103 / 80** | 84 / 72 | 93 / 78 | 95 / 76 |
| Position median, right / left | 4.8 / 4.3 mm | 3.2 / 3.0 | 3.3 / 3.3 | 4.7 / 4.5 | 3.0 / 2.9 |
| Battery posture failures | 2 | 7 | 35 | 22 | 20 |
| Missions, all three configs | — | 60/60 each | — | — | 60/60 each |

Later checkpoints place more tightly but arrive further from HOME. 2500 and 3999 tie on missions (60/60 in all three configs). 2500 wins on:
- **Arrival posture:** 3.6° vs 5.5° at the kick. 5.5° is already outside the ±0.08 rad (4.6°) start-stance range proposed for BallKick v2.
- **Battery:** the best of the five checkpoints.
- **Right-foot roll:** 1.21 m vs 1.09 m.

Lesson from v1: an arrival stance has to be the next policy's start stance.

## Run it (from the microduck_rl repo root)

```bash
V=logs/rsl_rl/velstand/2026-11-03_10-56-19_round3/velstand_round3_model_3999.onnx
KL=<ballkick_left_ts0p25_model_1250.onnx>
P=<precision_step_v2_model_2500.onnx>

# the regression geometry in the live viewer (left foot, +9 mm)
KICK_STANCE_SHIFT_LEFT=0.009 uv run python scripts/kick_mission.py $V $KL --foot left --precision $P --viewer --auto 30

# placement battery, split by ball side
uv run python scripts/step_battery.py $P --episodes 200 --prestand $V --seed 1 --ball-side both

# train it again
MICRODUCK_PSTEP_BALL_SIDES=both uv run train Mjlab-PrecisionStep-Flat-MicroDuck \
  --env.scene.num-envs 2048 --agent.max-iterations 4000 --video True
```

`kick_mission.py` is unchanged from [`../ballkick_left/code/`](../ballkick_left/code/).

## Files

| Path | What it is |
|---|---|
| `precision_step_v2_model_2500.onnx` | The policy |
| `metadata.json` | Hashes, contract, training settings, measured results |
| `config/` | Task config (with the env var), the run's `agent.yaml` / `env.yaml` (records `ball_left_prob: 0.5`), and the PrecisionStep extract of `mdp.py` |
| `code/` | `step_battery.py` (`--ball-side`), cfg tests (4 new), `verify_onnx.py` |
| `results/eval_2026-11-13/` | `SUMMARY.txt`, batteries for v1 and five v2 checkpoints, missions for 2500 and 3999 (right, left, left +9 mm) |
| `results/viewer_auto30_2026-11-13/` | Dillon's 30/30 viewer run at +9 mm: event logs and tally |

`code/` and `config/` are a snapshot of [dillonfaulk/microduck_rl @ local/velstand-own-teachers](https://github.com/dillonfaulk/microduck_rl/tree/local/velstand-own-teachers), commits `4999a477` (code) and `60b01d15` (the run, every checkpoint, all evals).

## Known limits

- **The left foot's +9 mm stance has no aim calibration yet.** It's now safe to use, but whether to use it, and its aim offset, are BallKick v2 calibration decisions.
- **The posture rule fails more often than v1's** (see the battery). No mission has failed because of it.
- **Kick direction now tracks sideways placement** (corr +0.6 in left-foot missions). The remaining spread comes from BallKick, which is what v2 of the kick targets.
- **Sim only, ground-truth pose and ball.**

## License and attribution

The policy file is CC-BY-NC-SA-4.0, matching the other Microduck policies here. The task is new in the fork, built on [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (Apache-2.0, see `LICENSE_UPSTREAM_APACHE-2.0`). Credit Pollen Robotics.
