from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.policies.base import PolicyResult, state_id_from
from src.policies.rule_based import run_rule_based_policy


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    batch_size = t.shape[0]
    if a.device != t.device:
        a = a.to(t.device)
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def _vp_beta_schedule(timesteps: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    t = np.arange(1, timesteps + 1)
    total = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = np.exp(-b_min / total - 0.5 * (b_max - b_min) * (2 * t - 1) / total**2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int = 8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, 2))

    def _compl_mul1d(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bix, iox -> box", a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, length = x.shape
        x_ft = torch.fft.rfft(x)
        modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        weight = torch.view_as_complex(self.weight[:, :, :modes])
        out_ft[:, :, :modes] = self._compl_mul1d(x_ft[:, :, :modes], weight)
        return torch.fft.irfft(out_ft, n=length)


class DiffFNO(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        width: int = 64,
        modes: int = 8,
        n_layers: int = 2,
        t_dim: int = 16,
        activation: str = "mish",
    ) -> None:
        super().__init__()
        act = nn.Mish if activation == "mish" else nn.ReLU

        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, width),
            act(),
            nn.Linear(width, width),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            act(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.cond_proj = nn.Linear(width + t_dim, width)
        self.input_proj = nn.Conv1d(1, width, kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv1d(width, width, modes=modes) for _ in range(n_layers)]
        )
        self.pointwise = nn.ModuleList([nn.Conv1d(width, width, kernel_size=1) for _ in range(n_layers)])
        self.activation = act()
        self.out_proj = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            act(),
            nn.Conv1d(width, 1, kernel_size=1),
        )
        self.residual = nn.Linear(action_dim, action_dim)
        self.residual_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        device = x.device
        if time.device != device:
            time = time.to(device)
        if state.device != device:
            state = state.to(device)

        state_feat = self.state_mlp(state)
        t_feat = self.time_mlp(time)
        cond = self.cond_proj(torch.cat([state_feat, t_feat], dim=-1)).unsqueeze(-1)

        y = self.input_proj(x.unsqueeze(1))
        y = y + cond

        for spec_conv, pointwise in zip(self.spectral_layers, self.pointwise):
            freq_out = spec_conv(y)
            point_out = pointwise(y)
            y = self.activation(freq_out + point_out + cond)

        out = self.out_proj(y).squeeze(1)
        res = self.residual(x)
        gate = torch.sigmoid(self.residual_gate)
        return out + gate * res


class DoubleCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        activation: str = "mish",
    ) -> None:
        super().__init__()
        act = nn.ReLU
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.q1_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        processed_state = self.state_mlp(state)
        x = torch.cat([processed_state, action], dim=-1)
        return self.q1_net(x), self.q2_net(x)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.forward(obs, action))


class Diffusion(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        model: nn.Module,
        max_action: float,
        beta_schedule: str = "vp",
        n_timesteps: int = 5,
        loss_type: str = "l2",
        clip_denoised: bool = True,
        bc_coef: bool = False,
        guidance_scale: float = 0.0,
        guidance_fn: Any | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model

        if beta_schedule != "vp":
            raise ValueError("Only vp beta schedule is supported by the local DROPT adapter.")
        betas = _vp_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.bc_coef = bc_coef
        self.guidance_scale = guidance_scale
        self.guidance_fn = guidance_fn

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if self.bc_coef:
            return noise
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))
        if self.guidance_fn is not None and self.guidance_scale > 0:
            x_recon_detached = x_recon.detach().requires_grad_(True)
            guidance = self.guidance_fn(x_recon_detached, s, t)
            if guidance is not None:
                x_recon = (x_recon_detached - self.guidance_scale * guidance).detach()
        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            raise RuntimeError("clip_denoised=False is not supported in the local DROPT adapter.")
        return self.q_posterior(x_start=x_recon, x_t=x, t=t)

    def p_sample(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(batch_size, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, state: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)
        return x

    def sample(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape)
        return action.clamp_(-self.max_action, self.max_action)

    def forward(self, state: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.sample(state, *args, **kwargs)


class DROPTPolicyBundle(nn.Module):
    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 6,
        hidden_dim: int = 256,
        fno_width: int = 48,
        fno_modes: int = 4,
        fno_layers: int = 1,
        t_dim: int = 16,
        activation: str = "mish",
        diffusion_steps: int = 6,
        max_action: float = 1.0,
    ) -> None:
        super().__init__()
        actor_model = DiffFNO(
            state_dim=state_dim,
            action_dim=action_dim,
            width=fno_width,
            modes=fno_modes,
            n_layers=fno_layers,
            t_dim=t_dim,
            activation=activation,
        )
        self._actor = Diffusion(
            state_dim=state_dim,
            action_dim=action_dim,
            model=actor_model,
            max_action=max_action,
            beta_schedule="vp",
            n_timesteps=diffusion_steps,
            bc_coef=True,
        )
        self._target_actor = Diffusion(
            state_dim=state_dim,
            action_dim=action_dim,
            model=DiffFNO(
                state_dim=state_dim,
                action_dim=action_dim,
                width=fno_width,
                modes=fno_modes,
                n_layers=fno_layers,
                t_dim=t_dim,
                activation=activation,
            ),
            max_action=max_action,
            beta_schedule="vp",
            n_timesteps=diffusion_steps,
            bc_coef=True,
        )
        self._critic = DoubleCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self._target_critic = DoubleCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)


class DROPTCheckpointPolicy:
    def __init__(
        self,
        model_path: str | Path | None,
        fallback_policy: Any | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.device = torch.device(device)
        self.fallback_policy = fallback_policy or run_rule_based_policy
        self._bundle: DROPTPolicyBundle | None = None

        if self.model_path is not None:
            self._bundle = self._load_bundle(self.model_path)

    def _load_bundle(self, path: Path) -> DROPTPolicyBundle:
        try:
            raw = torch.load(path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - exercised in tests
            raise RuntimeError(f"failed to load DROPT checkpoint from {path}: {exc}") from exc

        state_dict = self._extract_state_dict(raw)
        bundle = DROPTPolicyBundle().to(self.device)
        try:
            bundle.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise RuntimeError(f"failed to load DROPT checkpoint from {path}: {exc}") from exc
        bundle.eval()
        return bundle

    def run(self, state: dict[str, Any]) -> PolicyResult:
        input_state_id = state_id_from(state)
        if self._bundle is None:
            result = self.fallback_policy(state)
            return result.model_copy(
                update={
                    "policy_name": "dropt_checkpoint_fallback",
                    "baseline": "rule_based",
                    "notes": _append_note(result.notes, "DROPT checkpoint not configured; fell back to rule-based policy."),
                }
            )

        state_vector = _extract_bear_state_vector(state)
        if state_vector is None:
            result = self.fallback_policy(state)
            return result.model_copy(
                update={
                    "policy_name": "dropt_checkpoint_fallback",
                    "baseline": "rule_based",
                    "notes": _append_note(
                        result.notes,
                        "DROPT checkpoint requires an explicit 20-dimensional BEAR state vector; fell back to rule-based policy.",
                    ),
                }
            )

        action = self._predict_action(input_state_id, state_vector)
        mean_action_change = float(np.mean(np.abs(action)))
        return PolicyResult(
            policy_name="dropt_guided_diffno_checkpoint",
            input_state_id=input_state_id,
            recommended_action=action,
            estimated_energy=None,
            estimated_comfort_violations=None,
            mean_action_change=mean_action_change,
            baseline="guided_diffno",
            notes=(
                "Loaded local Guided-DiffFNO checkpoint and ran deterministic sampling on an explicit BEAR state vector."
            ),
        )

    def _predict_action(self, input_state_id: str, state_vector: list[float]) -> list[float]:
        assert self._bundle is not None
        state_tensor = torch.tensor([state_vector], dtype=torch.float32, device=self.device)
        seed = _stable_seed(input_state_id, state_vector)

        with torch.no_grad():
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                action_tensor = self._bundle._actor.sample(state_tensor)

        action = action_tensor.squeeze(0).detach().cpu().tolist()
        if not all(math.isfinite(value) for value in action):
            raise RuntimeError("DROPT checkpoint produced non-finite actions.")
        return [float(value) for value in action]

    @staticmethod
    def _extract_state_dict(raw: Any) -> dict[str, torch.Tensor]:
        if isinstance(raw, dict):
            for key in ("model", "state_dict", "policy", "checkpoint"):
                maybe = raw.get(key)
                if isinstance(maybe, dict):
                    return maybe
            if all(isinstance(value, torch.Tensor) for value in raw.values()):
                return raw
        if hasattr(raw, "keys") and all(isinstance(value, torch.Tensor) for value in raw.values()):
            return dict(raw)
        raise RuntimeError("failed to load DROPT checkpoint: unsupported checkpoint structure")


def _extract_bear_state_vector(state: dict[str, Any]) -> list[float] | None:
    for key in ("bear_state_vector", "bear_state", "state_vector", "observation", "obs", "raw_state"):
        candidate = state.get(key)
        if candidate is None:
            continue
        values = np.asarray(candidate, dtype=np.float32).reshape(-1)
        if values.size == 20 and np.isfinite(values).all():
            return [float(value) for value in values.tolist()]
    return None


def _stable_seed(input_state_id: str, state_vector: list[float]) -> int:
    vector_bytes = np.asarray(state_vector, dtype=np.float32).tobytes()
    digest = hashlib.sha256(input_state_id.encode("utf-8") + b"|" + vector_bytes).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _append_note(existing: str, extra: str) -> str:
    if existing:
        return f"{existing} {extra}"
    return extra
