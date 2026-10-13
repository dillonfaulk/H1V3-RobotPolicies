# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP (Adversarial Motion Priors) training components for asimov-1, ported from menloresearch/cyclotron's
official training repo for this robot. See amp_ppo.py for the algorithm, motion_dataset.py for the reference
clip loader, and ../amp_env_cfg.py for how this plugs into the velocity-command task."""
