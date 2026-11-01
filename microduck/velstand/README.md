# Microduck VelStand (mjlab) — walks and gets up, one policy

**Trained in mjlab (MuJoCo Warp), not Isaac Lab.** Sim-validated only; this policy has never run on the physical robot.

Microduck is Pollen Robotics' ~800 g open-source biped. VelStand is a single policy that follows velocity commands and gets back up after a fall. It was trained with [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) using our own teachers. The full code, every training run and the eval scripts are in [dillonfaulk/microduck_rl @ local/velstand-own-teachers](https://github.com/dillonfaulk/microduck_rl/tree/local/velstand-own-teachers).

## Files

| File | What it is |
|---|---|
| `velstand_remap_model_6750.onnx` | **Current policy** (round 2, iteration 6750) |
| `velstand_model_3000.onnx` | Predecessor (chunk 1, iteration 3000), kept for lineage |
| `metadata.json` | Contract, lineage, checksums, measured results, known weaknesses |
| `config/microduck_velstand_env_cfg.py`, `config/distill.py`, `config/microduck_standup_env_cfg.py` | Task config, teacher/distillation code and stand-teacher config, from commit `6bc72eba` |
| `config/round2_remap_run/`, `config/chunk1_run/` | `agent.yaml` (RSL-RL PPO config) and `env.yaml`, exactly as each run saved them |

## Contract

- Observation `[1, 61]` → action `[1, 14]`, 50 Hz (physics 5 ms × decimation 4).
- The observation normalizer is **baked into the ONNX** (exported with microduck_rl's `scripts/export.py`). Feed raw observations. Each ONNX was checked against its PyTorch checkpoint (max difference about 1e-6).
- The gravity input is **projected gravity**, not the raw accelerometer.

## Drive it (CPU MuJoCo viewer)

```bash
cd microduck_rl
uv run python scripts/infer_policy.py --new-cmd-obs \
  --scene src/mjlab_microduck/robot/microduck/scene_allcollisions.xml \
  --walking /path/to/velstand_remap_model_6750.onnx
```

The all-collisions scene is required: the policy trained with neck, head and upper-leg collisions. Don't use `logs/drive_remap.py` with model_6750. That wrapper was calibrated for model_3000 and now overshoots.

## How it was trained

1. **Walk anchor:** our own velocity walk (`model_4750`).
2. **Stand teacher:** our own StandUp expert (`supportgate/model_3500`). A support-contact gate pays the standing rewards only when nothing but the feet touches the floor, which closed a "head tripod" reward exploit.
3. **VelStand chunk 1:** 4,000 iterations at 2,048 envs, warm-started from the walk. Behavior cloning from the stand teacher on fallen frames and the walk anchor on upright frames → `model_3000`.
4. **Round 2:** 4,000 more iterations from `model_3000` → `model_6750`. The walk anchor had speed dead zones, and the student copied them. So during training the anchor was fed the forward, sideways and turn command that makes it reach the asked speed, from a lookup table calibrated on the anchor itself.

## Measured (model_6750)

- **CPU sim:** 46/48 fallen starts got up (left side 8/8). No falls in any walking test. Forward 0.15 → 0.15 m/s, 0.30 → 0.26; turn ±1.0 → 1.09 / −1.08 rad/s.
- **Training sim:** 99% prone get-ups. Forward speed tracking is nearly exact from 0.1 to 0.25 m/s.

## Known weaknesses

- **Violent get-ups**, about 14 rad/s peak trunk spin. They must soften before the real robot.
- 0.1 m/s forward or backward from standstill does nothing.
- A slow right turn (0.5 rad/s) does nothing; left works.
- Backing up curves slightly.
- Strafe tops out around 0.08 m/s, a limit of the walk teacher.

## License and attribution

Trained with [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) by Pollen Robotics. The upstream code is Apache-2.0, and the config snapshots in `config/` are derived from it and keep that license (see `LICENSE_UPSTREAM_APACHE-2.0`). Pollen's 3D mesh files (`.stl`) are CC-BY-NC-SA-4.0. None are included here, but the policies were trained in simulation on the robot those meshes describe.

**The policy files (`*.onnx`) are released under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).** That's my choice, matching the terms Pollen uses for the Microduck's 3D models; Pollen's licenses don't require it. Credit Pollen Robotics ([pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)) and this repo, use them non-commercially only, and share adaptations under the same license.
