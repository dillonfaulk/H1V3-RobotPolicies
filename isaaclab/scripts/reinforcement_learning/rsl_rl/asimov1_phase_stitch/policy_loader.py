"""Loads an asimov1 actor directly from its `actor.safetensors` for pure-inference deployment --
no RSL-RL `OnPolicyRunner`/checkpoint-dict machinery, no ONNX. Reuses the exact MLP-reconstruction
logic `H1V3-RobotPolicies/tools/export_onnx.py` already proved correct (`build_mlp`/`infer_dims`,
confirmed via that tool's own numerical cross-check against PyTorch at export time) -- ported
in-line here rather than imported, since that tool lives in a separate repo not on this project's
Python path.

Handles the one real asymmetry between the two policies this phase machine stitches together: the
fall-recovery actor was trained with `obs_normalization=True` (the only checkpoint in the whole
project trained this way) and saves real `obs_normalizer._mean`/`_std` tensors alongside the MLP
weights; the walking actor has no such tensors. `LoadedPolicy.act()` applies the normalizer only
when present, so a single interface serves both policies correctly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from safetensors.torch import load_file

ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


def _build_mlp(obs_dim: int, action_dim: int, hidden_dims: list[int], activation: str) -> nn.Sequential:
    """Mirrors export_onnx.py's build_mlp exactly: Linear+activation repeated, final Linear with no
    activation. nn.Sequential auto-names children "0","1","2",... matching the "mlp.0.weight"/
    "mlp.2.weight"/... keys in the checkpoint once the "mlp." prefix is stripped."""
    act_cls = ACTIVATIONS[activation]
    dims = [obs_dim] + hidden_dims
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(act_cls())
    layers.append(nn.Linear(dims[-1], action_dim))
    return nn.Sequential(*layers)


def _infer_dims(mlp_state: dict) -> tuple[int, int, list[int]]:
    """Mirrors export_onnx.py's infer_dims: derive obs_dim/action_dim/hidden_dims directly from the
    Linear layer shapes, no hardcoded per-robot assumption."""
    weight_items = sorted(
        ((k, v) for k, v in mlp_state.items() if k.endswith(".weight")),
        key=lambda kv: int(kv[0].split(".")[0]),
    )
    obs_dim = weight_items[0][1].shape[1]
    action_dim = weight_items[-1][1].shape[0]
    hidden_dims = [w.shape[0] for _, w in weight_items[:-1]]
    return obs_dim, action_dim, hidden_dims


class LoadedPolicy:
    """A deployable actor: the MLP, its action_scale, and (if present) its obs normalizer."""

    def __init__(self, mlp: nn.Module, action_scale: float, obs_mean: torch.Tensor | None, obs_std: torch.Tensor | None):
        self.mlp = mlp
        self.action_scale = action_scale
        self.obs_mean = obs_mean
        self.obs_std = obs_std

    @property
    def obs_dim(self) -> int:
        first_linear = self.mlp[0]
        return first_linear.in_features

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action mean (no exploration sampling) -- same deployment convention this
        project uses everywhere. Applies the saved normalizer first if this policy has one."""
        if self.obs_mean is not None:
            obs = (obs - self.obs_mean) / self.obs_std
        return self.mlp(obs)


def load_policy(
    safetensors_path: str, action_scale: float, device: str = "cpu", activation: str = "elu"
) -> LoadedPolicy:
    """Load one actor from its actor.safetensors file. `action_scale` must be passed explicitly --
    it's a task-config fact (0.25 for walking, 0.5 for recovery), not something recoverable from
    the tensor file itself. `device` must match the simulation env's own device (e.g. "cuda:0") --
    observations come off the env on that device, and PyTorch requires both operands of a matmul
    to live on the same device."""
    state = load_file(safetensors_path, device=device)
    mlp_state = {k[len("mlp."):]: v for k, v in state.items() if k.startswith("mlp.")}
    if not mlp_state:
        raise ValueError(f"no mlp.* tensors found in {safetensors_path}")

    obs_dim, action_dim, hidden_dims = _infer_dims(mlp_state)
    mlp = _build_mlp(obs_dim, action_dim, hidden_dims, activation).to(device)
    mlp.load_state_dict(mlp_state, strict=True)
    mlp.eval()

    obs_mean = state.get("obs_normalizer._mean")
    obs_std = state.get("obs_normalizer._std")

    return LoadedPolicy(mlp, action_scale, obs_mean, obs_std)
