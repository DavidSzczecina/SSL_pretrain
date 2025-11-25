# ssl_pretrainers.py
from __future__ import annotations
from typing import Dict, Callable, Optional, Tuple
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


SAVE_EPOCHS = {5, 10, 25, 50, 75, 100}

# --------- Minimal NT-Xent from your file ---------
class NTXent(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.t = temperature
    def forward(self, z_i, z_j):
        B, D = z_i.shape
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.matmul(z, z.T)
        mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -9e15)
        pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
        logits = sim / self.t
        labels = pos
        return F.cross_entropy(logits, labels)
# -----------------------------------------------

class BasePretrainer:
    """Interface: build(model/backbone -> projection wrapper?), fit(ssl_loader) -> encoder"""
    name: str = "base"
    def build(self, encoder_backbone: nn.Module, **hparams) -> nn.Module:
        raise NotImplementedError
    def fit(self, model: nn.Module, ssl_loader, device: str, **hparams) -> None:
        raise NotImplementedError
    def extract_encoder(self, model: nn.Module) -> nn.Module:
        """Return the (possibly updated) encoder backbone from the SSL model."""
        raise NotImplementedError

# ---------- SimCLR implementation ----------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        z = self.mlp(x)
        return F.normalize(z, dim=1)

class SimCLRWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, proj: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.proj = proj
    def forward(self, x):
        feat = self.encoder(x, return_feat=True) 
        return self.proj(feat)

class SimCLRPretrainer(BasePretrainer):
    name = "simclr"

    def build(self, encoder_backbone: nn.Module, in_dim=512, proj_dim=128, hidden=512, **_):
        proj = ProjectionHead(in_dim=in_dim, hidden_dim=hidden, out_dim=proj_dim)
        return SimCLRWrapper(encoder_backbone, proj)

    def fit(self, model: nn.Module, ssl_loader, device: str,
            epochs: int = 10, lr: float = 1e-3, wd: float = 1e-4,
            save_every: int = 5, save_fn=None, temperature: float = 0.5, logger=print):
        """
        Trains SimCLR with optional checkpoint saving every `save_every` epochs.
        """
        loss_fn = NTXent(temperature)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        t0 = time.time()
        model.to(device)

        for ep in range(1, epochs + 1):
            model.train()
            loss_sum, n = 0.0, 0

            for (x_i, x_j), _ in ssl_loader:
                x_i, x_j = x_i.to(device), x_j.to(device)

                opt.zero_grad(set_to_none=True)
                li = model(x_i)
                lj = model(x_j)
                loss = loss_fn(li, lj)
                loss.backward()
                opt.step()

                bs = x_i.size(0)
                loss_sum += loss.item() * bs
                n += bs

            logger(f"[SimCLR] Epoch {ep:03d} | Loss={loss_sum / max(n, 1):.4f}")

            #if save_every > 0 and save_fn is not None and ep % save_every == 0:
            if save_fn is not None and ep in SAVE_EPOCHS:
                save_fn(model, ep)

        logger(f"[SimCLR] Done in {(time.time() - t0) / 60:.2f} min")
        return model

    def extract_encoder(self, model: SimCLRWrapper) -> nn.Module:
        return model.encoder







#helpers
def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)

class _PredictorMLP(nn.Module):
    """2-layer predictor used by SimSiam/BYOL."""
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x): return self.net(x)

class _ProjMLP(nn.Module):
    """Projection head for methods that need plain MLP (use if ProjectionHead isn't available)."""
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False),  # as in SimSiam/BYOL heads
        )
    def forward(self, x): return self.net(x)

# If your file already defines ProjectionHead(in_dim, hidden_dim, out_dim) use it.
# We try to reuse it; otherwise fallback to _ProjMLP.
def _make_proj(in_dim, hidden_dim, out_dim):
    try:
        return ProjectionHead(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    except NameError:
        return _ProjMLP(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)




# Barlow Twins
def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class _BarlowNet(nn.Module):
    def __init__(self, encoder: nn.Module, proj: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.proj = proj
    def forward(self, x):
        h = self.encoder(x, return_feat=True)
        z = self.proj(h)
        # BN already applied inside proj; normalize to zero-mean across batch if you like
        return z

class BarlowTwinsPretrainer(BasePretrainer):
    name = "barlow"

    def build(self, encoder_backbone: nn.Module, in_dim=512, proj_dim=2048, hidden=2048, **_):
        proj = _make_proj(in_dim=in_dim, hidden_dim=hidden, out_dim=proj_dim)
        return _BarlowNet(encoder_backbone, proj)

    def fit(self, model: nn.Module, ssl_loader, device: str,
            epochs: int = 100, lr: float = 3e-4, wd: float = 1e-4,
            lambd: float = 5e-3,
            save_every: int = 5, save_fn=None, logger=print, **_):
        """
        Barlow Twins objective:
            L = sum_i (1 - C_ii)^2 + λ * sum_{i!=j} C_ij^2
        where C is cross-correlation between z1 and z2 across batch.
        """
        t0 = time.time()
        model.to(device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        for ep in range(1, epochs+1):
            ep_loss, n = 0.0, 0
            for (x1, x2), _ in ssl_loader:
                x1, x2 = x1.to(device), x2.to(device)
                z1 = model(x1)
                z2 = model(x2)

                z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-9)
                z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-9)

                N, D = z1.shape
                c = (z1.T @ z2) / (N - 1)

                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = _off_diagonal(c).pow_(2).sum()
                loss = on_diag + lambd * off_diag

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                ep_loss += loss.item() * x1.size(0)
                n += x1.size(0)

            logger(f"[Barlow] epoch {ep:03d} | loss {ep_loss/max(n,1):.4f}")

            #if save_every > 0 and save_fn is not None and ep % save_every == 0:
            if save_fn is not None and ep in SAVE_EPOCHS:
                save_fn(model, ep)

        logger(f"[Barlow] Done in {(time.time() - t0) / 60:.2f} min")
        return model

    def extract_encoder(self, model: nn.Module) -> nn.Module:
        return model.encoder


_PRETRAINERS: Dict[str, Callable[[], BasePretrainer]] = {
    SimCLRPretrainer.name: SimCLRPretrainer,
    BarlowTwinsPretrainer.name: BarlowTwinsPretrainer,
}




def get_pretrainer(name: str) -> BasePretrainer:
    name = name.lower()
    if name not in _PRETRAINERS:
        raise ValueError(f"Unknown pretrainer '{name}'. Options: {list(_PRETRAINERS)}")
    return _PRETRAINERS[name]()
