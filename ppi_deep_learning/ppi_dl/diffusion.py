"""Diffusion utilities from Notebook 04, packaged for reuse in Notebook 06.

Everything here is explained step by step in ``04_diffusion_models.ipynb``:
a linear-beta DDPM schedule, v-prediction training with label dropout for
classifier-free guidance, an EMA copy of the weights, and a DDIM sampler.
Any denoiser with signature ``model(x_t, t, c) -> v_hat`` works, where
``x_t`` is (B, 4886), ``t`` is (B,) integer steps and ``c`` is (B,) labels
in {0, 1, NULL=2}.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T = 1000
NULL = 2
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
abar = torch.cumprod(alphas, 0)


def timestep_embedding(t, dim=128):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))

    def forward(self, x):
        return x + self.net(x)


class MLPDenoiser(nn.Module):
    """The flat-vector denoiser of Notebook 04."""
    def __init__(self, d, hidden=512, n_blocks=3):
        super().__init__()
        self.inp = nn.Linear(d, hidden)
        self.t_mlp = nn.Sequential(nn.Linear(128, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.c_emb = nn.Embedding(3, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, d))

    def forward(self, x, t, c):
        h = self.inp(x) + self.t_mlp(timestep_embedding(t)) + self.c_emb(c)
        return self.out(self.blocks(h))


def train_diffusion(model, X, y, epochs=80, bs=64, lr=1e-3, p_uncond=0.1, ema_decay=0.999,
                    log_every=20, verbose=True):
    """v-prediction training; returns (EMA model, per-epoch losses)."""
    X, y = torch.as_tensor(X), torch.as_tensor(y)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ema = copy.deepcopy(model).eval()
    losses = []
    for ep in range(epochs):
        model.train()
        perm, run = torch.randperm(len(X)), []
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            x0, c = X[idx], y[idx].clone()
            c[torch.rand(len(c)) < p_uncond] = NULL
            t = torch.randint(0, T, (len(x0),))
            eps = torch.randn_like(x0)
            ab = abar[t].view(-1, 1)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            v = ab.sqrt() * eps - (1 - ab).sqrt() * x0
            loss = F.mse_loss(model(xt, t, c), v)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
            run.append(loss.item())
        sched.step()
        losses.append(float(np.mean(run)))
        if verbose and ep % log_every == 0:
            print(f"epoch {ep:3d}  loss {losses[-1]:.4f}")
    return ema, losses


@torch.no_grad()
def guided_eps(model, x, t, c, w=0.0):
    tt = torch.full((len(x),), t, dtype=torch.long)
    v = model(x, tt, c)
    if w != 0:
        v = (1 + w) * v - w * model(x, tt, torch.full_like(c, NULL))
    ab = abar[t]
    return (1 - ab).sqrt() * x + ab.sqrt() * v


@torch.no_grad()
def sample_ddim(model, c, d, w=0.0, n_steps=50, batch=400):
    """Deterministic DDIM sampling with classifier-free guidance strength ``w``."""
    model.eval()
    c = torch.as_tensor(c)
    out = []
    ts = torch.linspace(T - 1, 0, n_steps).long()
    for s in range(0, len(c), batch):
        cb = c[s:s + batch]
        x = torch.randn(len(cb), d)
        for i, t in enumerate(ts):
            e = guided_eps(model, x, int(t), cb, w)
            x0 = (x - (1 - abar[t]).sqrt() * e) / abar[t].sqrt()
            ab_next = abar[ts[i + 1]] if i + 1 < n_steps else torch.tensor(1.0)
            x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * e
        out.append(x)
    return torch.cat(out).numpy()
