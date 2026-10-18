# H1V3-RobotPolicies

<p align="center">
  <img src="images/asimov1.webp" alt="asimov-1 humanoid" width="45%" />
  <img src="images/v1x1.png" alt="v1x1 quadruped" width="45%" />
</p>

<p align="center">
  <img src="images/asimov1_teleop_demo.gif" alt="asimov-1 live ROS2 keyboard teleop demo" width="80%" />
  <br>
  <em>asimov-1 driven live via the ROS2 /cmd_vel teleop bridge in this repo - proof of concept.</em>
</p>

Reinforcement learning locomotion policies trained in [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab), covering three robots:

- **asimov-1** ([menloresearch/asimov-1](https://github.com/menloresearch/asimov-1)) - a 1.2m/35kg open-source humanoid. Trained with PPO + Adversarial Motion Priors (AMP) for omnidirectional walking (forward/backward/strafe/turn), with a live ROS2 teleop bridge for real-time keyboard control.
- **v1x1** - a custom quadruped. Trained with PPO for omnidirectional walking, also with a live ROS2 teleop bridge.
- **BDX-R** ([BDX-R/BDX-R-IsaacLab](https://github.com/BDX-R/BDX-R-IsaacLab)) - an open-source Disney BDX-droid-inspired biped, hardware target under $3000. Trained with plain PPO for omnidirectional velocity-command walking on flat ground.

This repo does **not** contain the Isaac Lab framework itself - only the project-specific files (task definitions, reward/termination shaping, robot asset configs, converted USD models, training/play/teleop scripts, and a handful of validated checkpoints) needed to drop back into a fresh Isaac Lab checkout.

**Every validated checkpoint ships in three formats**, so you don't need this repo's Isaac Lab setup just to use a trained policy:
- **`.pt`** - the original RSL-RL training checkpoint (actor + critic + optimizer state). Needed to resume training; also the only format to open with PyTorch directly. Pickle-based, so only ever load one from a source you trust.
- **`.safetensors`** - the same weights with zero code-execution risk on load, split into `actor.safetensors` (deployable policy only) and `full.safetensors` (everything, for inspection/continued training).
- **`.onnx`** - the policy exported as a portable inference graph. Runs in ONNX Runtime, TensorRT, on a Jetson, or anywhere else with no PyTorch or Isaac Lab installed. Self-contained single file, numerically verified against the original PyTorch output at export time.

See [Safetensors and ONNX conversions](#safetensors-and-onnx-conversions) below for exactly what's in each, and how to regenerate them with `tools/convert_checkpoint.py`/`tools/export_onnx.py`.

## Prerequisites

- **Isaac Lab**, checked out at tag **`v3.0.0-beta2.patch1`** (this is the exact version everything here was built and tested against - newer versions may have breaking API changes):
  ```bash
  git clone --branch v3.0.0-beta2.patch1 git@github.com:isaac-sim/IsaacLab.git
  ```
  Follow Isaac Lab's own install instructions to get it running (Isaac Sim + the IsaacLab Python environment).
- **ROS2 Jazzy** (only needed for the teleop bridge scripts):
  ```bash
  sudo apt install ros-jazzy-desktop ros-jazzy-teleop-twist-keyboard
  ```

## Restoring into a fresh Isaac Lab checkout

```bash
# from the root of a fresh IsaacLab clone:
rsync -a /path/to/H1V3-RobotPolicies/isaaclab/ ./
git apply /path/to/H1V3-RobotPolicies/isaaclab/patches/robot_registry_init.patch
```

The patch adds two export lines (`V1X1_CFG`, `ASIMOV1_CFG`) to `isaaclab_assets`'s own `__init__.pyi` stub files so both robots show up alongside Isaac Lab's built-in ones (`ANT_CFG`, etc.). If the patch doesn't apply cleanly against a different Isaac Lab version, just add the two lines by hand - see the patch file for exactly what's needed, it's trivial.

Checkpoints aren't wired into the `isaaclab/` tree (so you can put them wherever you like) - see `checkpoints/` below, and pass their path via `--checkpoint` on any of the play scripts.

**BDX-R restores differently** - unlike asimov-1/v1x1, it's NOT merged into the Isaac Lab checkout itself. It's a separate external project installed as an editable pip package:
```bash
./isaaclab.sh -p -m pip install -e /path/to/H1V3-RobotPolicies/bdxr/source_BDXR
```
Then run its scripts directly from wherever you put `bdxr/scripts_rsl_rl/` (they're self-contained, just need `cli_args.py` alongside `train.py`/`play.py`, same as they ship). See [BDX-R: training status](#bdx-r-training-status) below for the exact fixes/launch commands.

## Repo layout

```
isaaclab/                    # mirrors Isaac Lab's own directory structure exactly
  scripts/reinforcement_learning/rsl_rl/
    train_asimov1_direct.py              # standing/balance task trainer
    train_asimov1_velocity_amp_direct.py # velocity-command (walking) task trainer, PPO+AMP
    train_asimov1_velocity_direct.py     # velocity-command task trainer, plain PPO (no AMP)
    train_asimov1_fallrecovery_direct.py # fall-recovery (getup) task trainer, plain PPO
    play_asimov1_velocity_amp.py         # watch a checkpoint play back live (task's own random commands)
    play_asimov1_fallrecovery.py         # watch a fall-recovery checkpoint play back live
    play_asimov1_cmd_vel.py              # ROS2 /cmd_vel teleop bridge for asimov-1
    play_v1x1_cmd_vel.py                 # ROS2 /cmd_vel teleop bridge for v1x1
    play_asimov1_phase_stitch.py         # live walk->fall->recover->walk phase-machine demo, see below
    asimov1_phase_stitch/                # phase-machine package: gates, obs assembly, policy loading,
                                          # push testing, status/control HTTP server (used by the above)
  source/isaaclab_assets/isaaclab_assets/robots/
    asimov1.py                # ASIMOV1_CFG articulation config (actuator gains, joint limits, etc.)
    v1x1.py                   # V1X1_CFG articulation config
  source/isaaclab_assets/data/Robots/
    asimov1/                  # converted USD robot model (incl. the orange/charcoal paint job + colors)
    v1x1/                     # converted USD robot model
  source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/
    asimov1/                  # asimov-1's task definitions: env cfgs, mdp.py, AMP subpackage, agents/
    v1x1/                     # v1x1's task definitions: env cfgs, mdp.py, agents/
  patches/
    robot_registry_init.patch # the two __init__.pyi diffs described above

ros2_packages/
  v1x1_description/           # v1x1's own URDF/xacro, rviz configs, launch files (ROS2 package)
                               # NOTE: also lives at github.com/dillonfaulk/v1x1_description as its own repo,
                               # but the copy here includes a local fix (hardcoded mesh file paths updated
                               # from an old machine hostname) that wasn't committed/pushed there yet.

bdxr/                          # vendored from github.com/BDX-R/BDX-R-IsaacLab (MIT, see LICENSE_UPSTREAM),
                                # with our own compatibility fixes applied -- see training status section below
  source_BDXR/                 # the BDXR pip package: robot cfg, task env cfg, mdp, agents, URDF+mesh data
  scripts_rsl_rl/               # train.py, play.py, cli_args.py
  LICENSE_UPSTREAM              # upstream repo's MIT license (Kayden Knapik)

checkpoints/
  asimov1/                     # validated milestone checkpoints, see below
  v1x1/
  bdxr/

tools/
  convert_checkpoint.py        # .pt -> safetensors + metadata.json converter, see below
  export_onnx.py                # .pt -> onnx converter, see below
```

## Safetensors and ONNX conversions

Each checkpoint also has a same-named subfolder next to its `.pt` (e.g.
`checkpoints/asimov1/asimov1_omni_quality_run2_validated/`) containing:
- `actor.safetensors` - just the deployable policy network (no code, loads instantly, safe to open from an untrusted source - unlike `.pt`, which is pickle and can execute arbitrary code on load).
- `full.safetensors` - every tensor in the checkpoint (actor + critic + AMP discriminator where present), for continuing training or inspection.
- `actor.onnx` - the deployable policy exported as a portable ONNX graph (runs in ONNX Runtime, TensorRT, on a Jetson, etc. with no PyTorch/Isaac Lab installed at all). Exports only the deterministic action mean (no exploration noise/sampling) - the `distribution.std_param` training-only tensor is intentionally excluded, noted in metadata. Self-contained single file (no external-data sidecar), self-describing via embedded ONNX metadata (obs_dim/action_dim/hidden_dims/activation), and numerically verified against the original PyTorch forward pass at export time (see `metadata.json`'s `onnx_export` block for the measured max diff).
- `metadata.json` - iteration, tensor shapes, source file hash, obs/action dims, and the `onnx_export` block above. `command_ranges`/`control_hz`/`action_scale` are left `null` (task-config-specific, not auto-extracted - pull from the matching `*_env_cfg.py` in `isaaclab/` if needed). Every checkpoint except `asimov1_fallrecovery_validated` has no observation-normalization stats (`obs_normalization.type: "none"`, trained with `empirical_normalization=False`, raw observations feed the policy directly) - **`asimov1_fallrecovery_validated` is the one exception**, trained with `obs_normalization=True` (see its own `metadata.json` for why, and a real caveat: its `actor.onnx` does NOT have the normalizer baked in, so raw unnormalized observations will produce wrong actions on that one specifically - prefer `actor.safetensors` for it until that's fixed).

Regenerate with:
```bash
./isaaclab.sh -p tools/convert_checkpoint.py checkpoints/<robot>/<name>.pt \
    --out checkpoints/<robot>/<name> --robot <robot> --isaac-lab v3.0.0-beta2.patch1 --run <name>
./isaaclab.sh -p tools/export_onnx.py checkpoints/<robot>/<name>.pt \
    --out checkpoints/<robot>/<name> --robot <robot>
```

## asimov-1: training status

Gym ids: `Isaac-Asimov1-Stand-v0` (standing/balance, solved), `Isaac-Asimov1-Velocity-v0` (plain PPO walking), `Isaac-Asimov1-Velocity-AMP-v0` (PPO+AMP walking, the main line of work).

The AMP variant uses a real reference motion clip from Menlo's own training repo (`menloresearch/cyclotron`) - a discriminator learns to distinguish "real walking frame" from "policy-generated frame," and the policy is rewarded for fooling it, on top of the usual velocity-tracking task reward.

**Checkpoints included** (all for `Isaac-Asimov1-Velocity-AMP-v0`):
- `asimov1_omni_step3_validated.pt` / `asimov1_omni_step4_validated.pt` - successive checkpoints from a curriculum that gradually widened the commandable velocity range from forward-only up toward full omnidirectional, each validated over multiple training runs before widening further.
- `asimov1_omni_quality_run2_validated.pt` - the current best: same omnidirectional range, with an additional tuning pass for gait smoothness/accuracy (stronger action-smoothness and upright-orientation penalties, calibrated against this project's own logged reward magnitudes rather than copied blindly from any reference, plus new per-joint pose-tolerance, foot-stumble, and foot-contact-force-limit reward terms). This is the checkpoint that drove a live ROS2 keyboard teleop test successfully.

Run it live:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_asimov1_velocity_amp.py \
    --checkpoint /path/to/asimov1_omni_quality_run2_validated.pt --num_envs 16 --viz kit
```

Drive it with a keyboard:
```bash
# terminal 1 (source ROS2 Jazzy first):
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_asimov1_cmd_vel.py \
    --checkpoint /path/to/asimov1_omni_quality_run2_validated.pt --num_envs 1 --viz kit

# terminal 2 (source ROS2 Jazzy first):
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
asimov-1's real forward axis is local +X (matches ROS `Twist.linear.x` directly, no axis remapping needed).

**Known open items** (as of this checkpoint): the commandable velocity range is close to, but not yet at, the full range asimov-1's original creators (Menlo) report using (`lin_vel_x=+/-0.8`, `lin_vel_y=+/-0.6` m/s) - widening the remaining gap is a deliberately paused task in favor of a gait-quality tuning pass first. A sit-to-stand/chair policy is a reasonable next project once walking quality is fully settled (the fall-recovery/getup policy below is already done).

**Important note if touching this asset's visuals**: the robot's materials use Isaac Lab's native instancing (`instanceable=true`) for GPU efficiency across many cloned training environments. USD's instancing rules mean a shared `material:binding` pointing outside a prim's own instanceable subtree gets **silently dropped** at composition time - editing the shared `AsimovGreen`/`AsimovGrey` material definitions has zero effect on the render. Recoloring must be done via each mesh prim's own `displayColor` attribute instead (a plain per-mesh value, not a cross-prim binding, so it isn't subject to the same rule).

## asimov-1: fall-recovery (getup) training status

Gym id: `Isaac-Asimov1-FallRecovery-v0`. Separate task from standing/walking above - starts from a random fallen pose (full random SO(3) orientation, random position, small velocity jitter) and has to reach a genuine standing pose on its own: lying down -> sit upright -> kneel -> stand. Plain PPO, no AMP (no reference getup motion clip exists for this asset).

No stock Isaac Lab fall-recovery/getup template existed to build from (unlike walking, which had H1/G1/Spot precedent) - this was built from first principles plus literature (HumanUP arXiv:2502.12152, HoST arXiv:2502.08378) and, partway through, by directly comparing against a real working implementation: `openhorizon-labs/asimov-v1-policies`, a separate open-source project training the same getup task on the same robot in mjlab/MuJoCo Warp. Their design (an assist-force curriculum - an upward lift on the pelvis that decays as measured success rate rises, inspired by HoST's own "vertical pulling force" trick - plus their specific PPO hyperparameters) is the single biggest factor that got this checkpoint working, ported into Isaac Lab's own event-manager/wrench-composer APIs.

**17 training runs total**. Runs 1-13 (plain PPO, 1500 iterations each, fresh restarts) progressively diagnosed and fixed: an "A-pose" local optimum (robot using its head as a tripod leg, runs 1-7), a sitting-waypoint plateau that a hard hand-leverage exploit kept re-triggering (runs 8-10), and a dwell-too-long-while-seated problem (runs 11-13) - each fix verified via a dry-run -> headless smoke test -> 300-iteration headless check pipeline before trusting it on a real run, catching 2 real reward-hacking exploits and 2 structural reward-shape bugs along the way. Runs 14-17 switched to OpenHorizon-matched PPO settings (`entropy_coef` 0->0.005, `obs_normalization` False->True, `learning_rate` 5e-4->1e-3) plus a contact-pattern reward (`feet_only_contact`, rewards the specific pattern of "both feet touching, pelvis not" once near standing height) and a hand-pushoff bonus, then stayed on ONE resumed lineage across all 4 runs (14->15->16->17, 15996 total iterations, never restarted fresh) - this combination of a longer/higher-exploration training budget with the assist-force curriculum is what actually broke through: **success rate went from 0% (zero genuine successes across runs 1-13) to 89% by the end of run 17, with the assist-force curriculum fully decayed to 0.0 (the policy succeeds completely unassisted)**.

**Checkpoint included**:
- `asimov1_fallrecovery_validated.pt` - end of the runs14-17 lineage (iteration 15996), 89% success rate, fully unassisted. **This is the only checkpoint in the repo trained with `obs_normalization=True`** - see its own `metadata.json` and the safetensors/ONNX section above for what that means for deployment (short version: use `actor.safetensors`, not `actor.onnx`, until the normalizer is baked into the ONNX export too).

Run it live:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_asimov1_fallrecovery.py \
    --checkpoint /path/to/asimov1_fallrecovery_validated.pt --num_envs 16 --viz kit
```

**Known open items**: landings are still visibly a little rough (the training-time assist force gave a noticeable "moon gravity" boost once a robot got upright, which inflated the final pose's joint-deviation penalty - this mostly self-corrected as the assist decayed to 0 over the course of run 17, but final pose quality is still the roughest-looking part of a successful attempt). Not yet tested on real hardware or against the stricter multi-session visual-review bar other checkpoints in this repo were held to before being called "validated" - call this a strong training-time result, not a fully closed-out milestone yet.

## asimov-1: policy-stitching phase machine (walk -> fall -> recover -> walk)

Chains the walking and fall-recovery checkpoints into one continuous live loop in a single simulated robot: walk, get pushed over, detect the fall, recover, resume walking. A sensor-gated state machine (not a learned transition), mirroring the pattern `openhorizon-labs/asimov-v1-policies` (a separate project, same robot) proved in sim.

Run it live:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_asimov1_phase_stitch.py \
    --walk_checkpoint /path/to/asimov1_omni_quality_run2_validated/actor.safetensors \
    --recovery_checkpoint /path/to/asimov1_fallrecovery_validated/actor.safetensors \
    --status_port 8800 --auto_push_interval_s 20 --viz kit
```
Status/control while running: `curl localhost:8800/status`, `curl -X POST localhost:8800/control/push`, `curl -X POST localhost:8800/control/reset`.

**Status**: working well, not perfect - roughly 3-4 clean recover-to-walk cycles out of every 5 attempts, each running well beyond the 4s no-refall success bar once stable. Two real bugs were found and fixed via direct live testing before reaching this point: the walking policy's real action scale is 0.5 (not 0.25, a different policy's value from the design brief this was built from), and an action-blend added at the recovery-to-walk handoff turned out to be actively harmful (averaging a standing-pose target with a gait-pose target produces a weak, indecisive intermediate command) - removing it roughly doubled the success rate. The training-time gravity-lift assist mechanism (environment-level, not part of either checkpoint) is also explicitly forced off for this demo, since it isn't part of either policy's real validated behavior.

**Known open items**: the remaining ~1-in-5 failures are suspected to be leftover joint velocity at the handoff moment (only root angular velocity is currently checked before handing off) and/or too aggressive an initial walk command right after recovery - both flagged as cheap next things to try before considering retraining either policy with more starting-state domain randomization.

## v1x1: training status

Gym ids: `Isaac-V1X1-v0` (single-vector walk toward a fixed target, solved), `Isaac-V1X1-Velocity-v0` (omnidirectional velocity-command walking, solved, drives the teleop bridge below).

v1x1 uses effort/torque control (`JointEffortActionCfg`) rather than position control, and trains with plain PPO (no AMP/reference motion).

**Checkpoints included**:
- `v1x1_SVwalk_final.pt` - the single-vector "walk toward a point" policy.
- `v1x1_omni_final.pt` - the omnidirectional velocity-command policy. This is the one the teleop bridge below is built for.

Drive it with a keyboard:
```bash
# terminal 1 (source ROS2 Jazzy first):
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_v1x1_cmd_vel.py \
    --checkpoint /path/to/v1x1_omni_final.pt --num_envs 1

# terminal 2 (source ROS2 Jazzy first):
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

**Important axis quirk**: v1x1's real anatomical forward is local **+Y**, not +X (a quirk of this robot's own mount orientation) - `play_v1x1_cmd_vel.py` already handles the remap internally (`Twist.linear.x` -> v1x1's `lin_vel_y` slot, `Twist.linear.y` -> `lin_vel_x`), so standard keyboard teleop still feels correct (forward key drives it forward) - just don't assume the raw command-vector axes line up with ROS convention if you're reading `vel_command_b` directly elsewhere.

## BDX-R: training status

Gym id: `Bdxr-Velocity-Flat-v0` (train), `Bdxr-Velocity-Flat-Play-v0` (play). Built on Isaac Lab's own stock
`LocomotionVelocityRoughEnvCfg` base class (same one H1/G1 use), trained with plain PPO (no AMP/reference
motion). Terrain switched from the upstream repo's own multi-row rough-terrain generator to a flat plane, to
match asimov-1/v1x1's own setup (same goal: walking, not rough-terrain generalization yet) and to spawn
environments in a tight, easy-to-view grid near the origin instead of scattered across a huge tiled map.

**Checkpoints included**:
- `bdxr_velocity_flat_validated.pt` - 1500 iterations from scratch, then resumed for 1500 more (2999 total) as
  a cleanup pass. Final: 100% success rate, ~84%/16% time_out/base_contact termination split, episode length
  ~930/1000 steps, velocity tracking error down to ~0.26 (xy) / ~0.24 (yaw).

Run it live:
```bash
./isaaclab.sh -p bdxr/scripts_rsl_rl/play.py \
    --task Bdxr-Velocity-Flat-Play-v0 --checkpoint /path/to/bdxr_velocity_flat_validated.pt --num_envs 50 --viz kit
```

**This upstream repo was written against a different Isaac Lab/rsl-rl-lib version than this repo is pinned
to** - getting it running against `v3.0.0-beta2.patch1` + the installed `rsl-rl-lib==5.0.1` required 6 real
compatibility fixes (all already applied in `bdxr/source_BDXR` and `bdxr/scripts_rsl_rl`, confirmed via direct
inspection, not guessed):
1. `isaaclab.utils.noise.AdditiveUniformNoiseCfg` renamed to `UniformNoiseCfg` in this pin.
2. `mdp.imu_projected_gravity` doesn't exist in this pin (no orientation on the `Imu` sensor data at all) -
   substituted the stock `mdp.projected_gravity`, which is exact here since this robot's IMU mount has a zero
   rotational offset.
3. A critic-only (not policy-facing) observation depending on a `JointWrenchSensor` type unavailable in this
   pin was dropped - zero effect on the deployed policy.
4. The `height_scanner`/`imu` sensors' hardcoded prim paths were missing a `/Geometry/` path segment that
   this URDF's on-the-fly conversion actually produces.
5. **The important one**: Isaac Lab's own `activate_contact_sensors()` helper only activates the PhysX
   contact-report API on the first rigid body it finds per branch - this URDF nests each leg link as a child
   of its parent link (mirroring the kinematic chain), so only `base_link` (1 of 11 bodies) was actually
   getting contact-sensed. Fixed by wrapping the robot's spawn function to re-walk the full tree and activate
   every rigid body, not just the first one - without this, every foot-contact-dependent reward term
   (air time, foot clearance, foot slip) would have had zero signal for 10 of 11 bodies.
6. The installed `rsl-rl-lib` (5.0.1) is two majors ahead of what `train.py`/`play.py` check for
   (`>=3.0.1`) - `isaaclab_rl.rsl_rl`'s own model-cfg classes leak deprecated fields into the actual model
   constructor unless `handle_deprecated_rsl_rl_cfg()` is called first, which neither script did by default.
   Patched both scripts to call it. **This would affect asimov-1/v1x1 too if either were retrained fresh
   against this same installed rsl-rl-lib version** - not just a BDX-R-specific issue.

## A note on scope

This repo captures the training pipeline and validated checkpoints at a point in time - it's a backup/continuity snapshot, not a polished public release. Config comments throughout the included files document a lot of the reasoning (and dead ends) behind specific numeric choices; they're left in deliberately since they're often more useful than the code alone when picking this back up later.
