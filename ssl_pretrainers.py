# ssl_pretrainers.py
from __future__ import annotations
from typing import Dict, Callable, Optional, Tuple
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



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

            if save_every > 0 and save_fn is not None and ep % save_every == 0:
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



# SimSiam
class _SimSiamNet(nn.Module):
    def __init__(self, encoder: nn.Module, proj: nn.Module, pred: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.proj = proj
        self.pred = pred  # predictor
    def forward(self, x):
        h = self.encoder(x, return_feat=True)
        z = self.proj(h)
        p = self.pred(z)
        return z, p

def _neg_cosine(p, z):
    # Stop-grad on z
    z = z.detach()
    p = _normalize(p)
    z = _normalize(z)
    return - (p * z).sum(dim=1).mean()

class SimSiamPretrainer(BasePretrainer):
    name = "simsiam"

    def build(self, encoder_backbone: nn.Module, in_dim=512, proj_dim=2048, hidden=2048, pred_dim=512, **_):
        proj = _make_proj(in_dim=in_dim, hidden_dim=hidden, out_dim=proj_dim)
        pred = _PredictorMLP(in_dim=proj_dim, hidden_dim=pred_dim, out_dim=proj_dim)
        return _SimSiamNet(encoder_backbone, proj, pred)

    def fit(self, model: nn.Module, ssl_loader, device: str,
            epochs: int = 100, lr: float = 3e-4, wd: float = 1e-4,
            save_every: int = 5, save_fn=None, logger=print, **_):
        t0 = time.time()
        model.to(device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        for ep in range(1, epochs+1):
            ep_loss, n = 0.0, 0
            for (x1, x2), _ in ssl_loader:
                x1, x2 = x1.to(device), x2.to(device)

                z1, p1 = model(x1)
                z2, p2 = model(x2)

                loss = _neg_cosine(p1, z2) / 2.0 + _neg_cosine(p2, z1) / 2.0
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                ep_loss += loss.item() * x1.size(0)
                n += x1.size(0)

            logger(f"[SimSiam] epoch {ep:03d} | loss {ep_loss/max(n,1):.4f}")

            if save_every > 0 and save_fn is not None and ep % save_every == 0:
                save_fn(model, ep)

        logger(f"[SimSiam] Done in {(time.time() - t0) / 60:.2f} min")
        return model

    def extract_encoder(self, model: nn.Module) -> nn.Module:
        return model.encoder

# BYOL
class _BYOLNet(nn.Module):
    """Online encoder+proj+pred and EMA target encoder+proj."""
    def __init__(self, online_enc: nn.Module, online_proj: nn.Module, predictor: nn.Module,
                 target_enc: nn.Module, target_proj: nn.Module):
        super().__init__()
        self.online_enc = online_enc
        self.online_proj = online_proj
        self.predictor = predictor

        self.target_enc = target_enc
        self.target_proj = target_proj
        for p in self.target_enc.parameters(): p.requires_grad = False
        for p in self.target_proj.parameters(): p.requires_grad = False

    @torch.no_grad()
    def update_target(self, m: float):
        # EMA update
        for p_o, p_t in zip(self.online_enc.parameters(), self.target_enc.parameters()):
            p_t.data = p_t.data * m + p_o.data * (1.0 - m)
        for p_o, p_t in zip(self.online_proj.parameters(), self.target_proj.parameters()):
            p_t.data = p_t.data * m + p_o.data * (1.0 - m)

    def online(self, x):
        h = self.online_enc(x, return_feat=True)
        z = self.online_proj(h)
        p = self.predictor(z)
        return p

    @torch.no_grad()
    def target(self, x):
        h = self.target_enc(x, return_feat=True)
        z = self.target_proj(h)
        return z.detach()

def _byol_loss(p, z):
    # l2-normalized cosine similarity loss (MSE between normalized vectors up to a constant)
    p = _normalize(p)
    z = _normalize(z)
    return 2 - 2 * (p * z).sum(dim=1).mean()

class BYOLPretrainer(BasePretrainer):
    name = "byol"

    def build(self, encoder_backbone: nn.Module, in_dim=512, proj_dim=256, hidden=4096, pred_dim=4096, **_):
        online_proj = _make_proj(in_dim=in_dim, hidden_dim=hidden, out_dim=proj_dim)
        predictor = _PredictorMLP(in_dim=proj_dim, hidden_dim=pred_dim, out_dim=proj_dim)

        # Make a deep copy for target encoder and head
        import copy
        target_enc = copy.deepcopy(encoder_backbone)
        target_proj = copy.deepcopy(online_proj)

        return _BYOLNet(encoder_backbone, online_proj, predictor, target_enc, target_proj)

    def fit(self, model: nn.Module, ssl_loader, device: str,
            epochs: int = 100, lr: float = 3e-4, wd: float = 1e-4,
            base_momentum: float = 0.99,
            save_every: int = 5, save_fn=None, logger=print, **_):
        t0 = time.time()
        model.to(device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        total_steps = epochs * max(1, len(ssl_loader))

        def momentum_schedule(step):
            if total_steps <= 1: return 1.0
            return 1.0 - (1.0 - base_momentum) * (0.5 * (1 + math.cos(math.pi * step / (total_steps-1))))

        step = 0
        for ep in range(1, epochs+1):
            ep_loss, n = 0.0, 0
            for (x1, x2), _ in ssl_loader:
                x1, x2 = x1.to(device), x2.to(device)

                p1 = model.online(x1)
                p2 = model.online(x2)
                with torch.no_grad():
                    z2 = model.target(x2)
                    z1 = model.target(x1)

                loss = (_byol_loss(p1, z2) + _byol_loss(p2, z1)) * 0.5

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                m = momentum_schedule(step); step += 1
                with torch.no_grad():
                    model.update_target(m)

                ep_loss += loss.item() * x1.size(0)
                n += x1.size(0)

            logger(f"[BYOL] epoch {ep:03d} | loss {ep_loss/max(n,1):.4f}")

            if save_every > 0 and save_fn is not None and ep % save_every == 0:
                save_fn(model, ep)

        logger(f"[BYOL] Done in {(time.time() - t0) / 60:.2f} min")
        return model

    def extract_encoder(self, model: nn.Module) -> nn.Module:
        return model.online_enc


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

            if save_every > 0 and save_fn is not None and ep % save_every == 0:
                save_fn(model, ep)

        logger(f"[Barlow] Done in {(time.time() - t0) / 60:.2f} min")
        return model

    def extract_encoder(self, model: nn.Module) -> nn.Module:
        return model.encoder


_PRETRAINERS: Dict[str, Callable[[], BasePretrainer]] = {
    SimCLRPretrainer.name: SimCLRPretrainer,
    SimSiamPretrainer.name: SimSiamPretrainer,
    BYOLPretrainer.name: BYOLPretrainer,
    BarlowTwinsPretrainer.name: BarlowTwinsPretrainer,
}




def get_pretrainer(name: str) -> BasePretrainer:
    name = name.lower()
    if name not in _PRETRAINERS:
        raise ValueError(f"Unknown pretrainer '{name}'. Options: {list(_PRETRAINERS)}")
    return _PRETRAINERS[name]()
