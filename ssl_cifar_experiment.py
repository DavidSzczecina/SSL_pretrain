#!/usr/bin/env python3
"""
CIFAR SimCLR vs Supervised — Noisy-Label Robustness Experiment (PyTorch)

Usage examples
--------------
# Baseline supervised on CIFAR-10 (clean)
python ssl_cifar_experiment.py --dataset cifar10 --mode baseline --epochs 100 --device cuda

# SimCLR pretrain on CIFAR-10 (unlabeled) then fine-tune supervised (clean labels)
python ssl_cifar_experiment.py --dataset cifar10 --mode simclr_then_finetune --pretrain-epochs 200 --epochs 100 --device cuda

# Baseline with 40% symmetric label noise
python ssl_cifar_experiment.py --dataset cifar10 --mode baseline --noise-rate 0.40 --epochs 100 --device cuda

# Fine-tune from a previously saved SimCLR encoder with 60% label noise
python ssl_cifar_experiment.py --dataset cifar10 --mode finetune_from_pretrained \
  --pretrained-encoder-path runs/checkpoints/simclr_cifar10_encoder.pt \
  --noise-rate 0.60 --epochs 100 --device cuda

Outputs
-------
- runs/metrics.csv: rows with settings + test accuracy
- runs/checkpoints/: model checkpoints + saved pretrained encoder
- runs/logs.txt: simple text log
"""

from __future__ import annotations
import os
import csv
import time
import math
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

# --------------------
# Utils
# --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def log_print(msg: str, log_path: Optional[str] = None):
    print(msg, flush=True)
    if log_path:
        with open(log_path, "a") as f:
            f.write(msg + "\n")

# --------------------
# Data & Augmentations
# --------------------
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

def get_cifar_norm(dataset: str):
    d = dataset.lower()
    if d == "cifar10":
        return CIFAR10_MEAN, CIFAR10_STD, 10
    elif d == "cifar100":
        return CIFAR100_MEAN, CIFAR100_STD, 100
    else:
        raise ValueError("dataset must be 'cifar10' or 'cifar100'")

class SimCLRAugmentCIFAR:
    """SimCLR-style augmentations for 32x32 color images (no normalization here)."""
    def __init__(self, size=32):
        self.base = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])
    def __call__(self, img):
        return self.base(img), self.base(img)

class NoisyLabelWrapper(Dataset):
    """Symmetric label noise wrapper applied on top of a base dataset with targets."""
    def __init__(self, base: Dataset, noise_rate: float, num_classes: int, seed: int = 42):
        self.base = base
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        g = random.Random(seed)
        # precompute noisy targets
        self.targets = []
        for _, y in base:
            if g.random() < noise_rate:
                ny = g.randrange(num_classes)
                if ny == y:
                    ny = (ny + 1) % num_classes
                self.targets.append(ny)
            else:
                self.targets.append(y)

    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, _ = self.base[i]
        return x, self.targets[i]

def get_dataloaders(dataset: str, batch_size: int, noise_rate: float,
                    num_workers: int, seed: int, for_simclr: bool):
    mean, std, num_classes = get_cifar_norm(dataset)
    sup_train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    root = os.path.join("data", dataset.lower())
    Base = datasets.CIFAR10 if dataset.lower() == "cifar10" else datasets.CIFAR100

    base_train = Base(root=root, train=True, download=True, transform=sup_train_tf)
    test = Base(root=root, train=False, download=True, transform=test_tf)

    if noise_rate > 0:
        train = NoisyLabelWrapper(base_train, noise_rate=noise_rate, num_classes=num_classes, seed=seed)
    else:
        train = base_train

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    ssl_ds = None
    if for_simclr:
        # Important: SSL pipeline should not include normalization
        tf_ssl = SimCLRAugmentCIFAR(size=32)
        ssl_ds = Base(root=root, train=True, download=True, transform=tf_ssl)

    return train_loader, test_loader, ssl_ds, num_classes

def build_ssl_loader(dataset: str, batch_size: int, num_workers: int) -> DataLoader:
    tf_ssl = SimCLRAugmentCIFAR(size=32)
    root = os.path.join("data", dataset.lower())
    Base = datasets.CIFAR10 if dataset.lower() == "cifar10" else datasets.CIFAR100
    ds = Base(root=root, train=True, download=True, transform=tf_ssl)

    class _Wrap(Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            (xi, xj), _ = self.ds[idx]
            return (xi, xj), 0

    ssl = _Wrap(ds)
    return DataLoader(ssl, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      drop_last=True, pin_memory=True)

# --------------------
# Models
# --------------------
class ResNet18Small(nn.Module):
    """
    ResNet-18 adapted for small 32x32 images.
    - conv1: kernel=3, stride=1, padding=1
    - remove initial maxpool
    - in_channels=3 for CIFAR
    """
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        self.features = nn.Sequential(*list(base.children())[:-1])  # until global avgpool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_feat: bool = False):
        x = self.features(x)
        x = torch.flatten(x, 1)
        if return_feat:
            return x
        return self.fc(x)

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
        z = F.normalize(z, dim=1)
        return z


class SimCLR(nn.Module):
    def __init__(self, encoder: ResNet18Small, proj: ProjectionHead):
        super().__init__()
        self.encoder = encoder
        self.proj = proj
    def forward(self, x):
        feat = self.encoder(x, return_feat=True)
        z = self.proj(feat)
        return z

# --------------------
# Loss
# --------------------
class NTXent(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.t = temperature

    def forward(self, z_i, z_j):
        B, D = z_i.shape
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        sim = torch.matmul(z, z.T)        # cosine similarity since z's are normalized
        mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -9e15)
        pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
        logits = sim / self.t
        labels = pos
        loss = F.cross_entropy(logits, labels)
        return loss

# --------------------
# Train / Eval
# --------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return correct / total, loss_sum / total

def train_supervised(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                     device: str, epochs: int, lr: float, wd: float, log: Optional[str] = None,
                     save_path: Optional[str] = None):
    start_time = time.time()  # <---- add this
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(1, epochs+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
        acc, tloss = evaluate(model, test_loader, device)
        log_print(f"[Supervised] Epoch {ep:03d} | Test Acc={acc*100:.2f}% | Test Loss={tloss:.4f}", log)
        if save_path and acc > best:
            best = acc
            torch.save(model.state_dict(), save_path)
    end_time = time.time()  # <---- add this
    log_print(f"[Supervised] Training finished in {(end_time - start_time)/60:.2f} minutes", log)
    return best

def pretrain_simclr(ssl_model: SimCLR, ssl_loader: DataLoader, device: str, epochs: int,
                    lr: float, wd: float, temperature: float, log: Optional[str] = None):
    start_time = time.time()  # <---- add this
    loss_fn = NTXent(temperature=temperature)
    opt = torch.optim.Adam(ssl_model.parameters(), lr=lr, weight_decay=wd)
    for ep in range(1, epochs+1):
        ssl_model.train()
        loss_sum, n = 0.0, 0
        for (x_i, x_j), _ in ssl_loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            opt.zero_grad()
            z_i = ssl_model(x_i)
            z_j = ssl_model(x_j)
            loss = loss_fn(z_i, z_j)
            loss.backward()
            opt.step()
            bs = x_i.size(0)
            loss_sum += loss.item() * bs
            n += bs
        log_print(f"[SimCLR] Epoch {ep:03d} | Loss={loss_sum/max(n,1):.4f}", log)
    end_time = time.time()  # <---- add this
    log_print(f"[SimCLR] Pretraining finished in {(end_time - start_time)/60:.2f} minutes", log)

# --------------------
# Orchestration
# --------------------
@dataclass
class RunResult:
    mode: str
    dataset: str
    noise_rate: float
    pretrain_epochs: int
    finetune_epochs: int
    test_acc: float
    timestamp: float

def save_metrics_row(path_csv: str, row: RunResult):
    ensure_dir(os.path.dirname(path_csv))
    header = list(asdict(row).keys())
    exists = os.path.exists(path_csv)
    with open(path_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow(asdict(row))

def save_encoder(encoder: nn.Module, path: str):
    ensure_dir(os.path.dirname(path))
    torch.save({"features": encoder.features.state_dict()}, path)

def load_encoder_into_classifier(path: str, clf: nn.Module):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("features", ckpt)
    missing, unexpected = clf.features.load_state_dict(state, strict=False)
    print(f"Loaded encoder from {path}. Missing={missing}, Unexpected={unexpected}")
    return clf

def main():
    p = argparse.ArgumentParser(description="CIFAR: SimCLR vs Supervised under Label Noise")
    p.add_argument("--exp-name", type=str, default="", 
               help="Experiment name for organizing results. "
                    "If not provided, auto-generated from mode/dataset/noise.")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10","cifar100"])
    p.add_argument("--mode", type=str, default="baseline",
                   choices=["baseline","simclr_then_finetune","finetune_from_pretrained"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10, help="supervised epochs (baseline or fine-tune)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--noise-rate", type=float, default=0.0, help="symmetric label noise in supervised training")
    # SSL pretrain
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--pretrain-wd", type=float, default=1e-4)
    p.add_argument("--batch-size-ssl", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--freeze-backbone", action="store_true", help="freeze encoder during fine-tuning")
    # Reuse         
    p.add_argument("--save-pretrained-encoder", type=str, default="simclr_cifar_encoder.pt")
    p.add_argument("--pretrained-encoder-path", type=str, default="")
    args = p.parse_args()

    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"


    # Directory setup
    base_results_dir = ("results_" + args.dataset)
    pretrained_dir = "pretrained_encoders"

    # Default experiment name if none provided
    if not args.exp_name:
        args.exp_name = f"{args.mode}_{args.dataset}_noise-{args.noise_rate}"

    exp_dir = os.path.join(base_results_dir, args.exp_name)
    ensure_dir(base_results_dir)
    ensure_dir(exp_dir)
    ensure_dir(pretrained_dir)

    # Paths for logging and metrics
    log_path = os.path.join(exp_dir, "logs.txt")
    metrics_path = os.path.join(exp_dir, "metrics.csv")


    # Supervised data (label noise applied here only)
    train_loader, test_loader, _, num_classes = get_dataloaders(
        dataset=args.dataset, batch_size=args.batch_size, noise_rate=args.noise_rate,
        num_workers=args.workers, seed=args.seed, for_simclr=False
    )

    if args.mode == "baseline":
        model = ResNet18Small(num_classes=num_classes, in_channels=3).to(device)
        ckpt = os.path.join(exp_dir, f"baseline_{args.dataset}_noise{args.noise_rate:.2f}.pt")
        acc = train_supervised(model, train_loader, test_loader, device,
                               epochs=args.epochs, lr=args.lr, wd=args.wd,
                               log=log_path, save_path=ckpt)
        log_print(f"Final Test Acc (baseline, {args.dataset}, noise={args.noise_rate:.2f}): {acc*100:.2f}%", log_path)
        save_metrics_row(metrics_path, RunResult(
            mode="baseline", dataset=args.dataset, noise_rate=args.noise_rate,
            pretrain_epochs=0, finetune_epochs=args.epochs, test_acc=acc, timestamp=time.time()
        ))
        return

    if args.mode == "simclr_then_finetune":
        # Pretrain (unlabeled) on same training images
        ssl_loader = build_ssl_loader(args.dataset, args.batch_size_ssl, args.workers)
        encoder = ResNet18Small(num_classes=num_classes, in_channels=3).to(device)
        proj = ProjectionHead(in_dim=512, hidden_dim=512, out_dim=args.proj_dim).to(device)
        ssl_model = SimCLR(encoder, proj).to(device)
        pretrain_simclr(ssl_model, ssl_loader, device,
                        epochs=args.pretrain_epochs, lr=args.pretrain_lr,
                        wd=args.pretrain_wd, temperature=args.temperature, log=log_path)
        # Save encoder for reuse
        if args.save_pretrained_encoder:
            save_encoder(encoder, os.path.join(pretrained_dir, args.save_pretrained_encoder))
            log_print(f"Saved pretrained encoder to: {os.path.join(pretrained_dir, args.save_pretrained_encoder)}", log_path)
        # Fine-tune
        clf = ResNet18Small(num_classes=num_classes, in_channels=3).to(device)
        clf.features.load_state_dict(encoder.features.state_dict())
        if args.freeze_backbone:
            for p in clf.features.parameters():
                p.requires_grad = False
        ckpt = os.path.join(exp_dir,
                            f"simclr_{args.dataset}_noise{args.noise_rate:.2f}_freeze{int(args.freeze_backbone)}.pt")
        acc = train_supervised(clf, train_loader, test_loader, device,
                               epochs=args.epochs, lr=args.lr, wd=args.wd,
                               log=log_path, save_path=ckpt)
        log_print(f"Final Test Acc (SimCLR→FT, {args.dataset}, noise={args.noise_rate:.2f}, freeze={args.freeze_backbone}): {acc*100:.2f}%", log_path)
        save_metrics_row(metrics_path, RunResult(
            mode=f"simclr_then_finetune_freeze{int(args.freeze_backbone)}", dataset=args.dataset, noise_rate=args.noise_rate,
            pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.epochs, test_acc=acc, timestamp=time.time()
        ))
        return

    if args.mode == "finetune_from_pretrained":
        if not args.pretrained_encoder_path or not os.path.exists(args.pretrained_encoder_path):
            raise FileNotFoundError("Set --pretrained-encoder-path to a valid .pt file saved from SimCLR pretrain.")
        clf = ResNet18Small(num_classes=num_classes, in_channels=3).to(device)
        load_encoder_into_classifier(args.pretrained_encoder_path, clf)
        if args.freeze_backbone:
            for p in clf.features.parameters():
                p.requires_grad = False
        ckpt = os.path.join(exp_dir,
                            f"ft_from_pretrained_{args.dataset}_noise{args.noise_rate:.2f}_freeze{int(args.freeze_backbone)}.pt")
        acc = train_supervised(clf, train_loader, test_loader, device,
                               epochs=args.epochs, lr=args.lr, wd=args.wd,
                               log=log_path, save_path=ckpt)
        log_print(f"Final Test Acc (FT from pretrained, {args.dataset}, noise={args.noise_rate:.2f}, freeze={args.freeze_backbone}): {acc*100:.2f}%", log_path)
        save_metrics_row(metrics_path, RunResult(
            mode=f"finetune_from_pretrained_freeze{int(args.freeze_backbone)}", dataset=args.dataset, noise_rate=args.noise_rate,
            pretrain_epochs=0, finetune_epochs=args.epochs, test_acc=acc, timestamp=time.time()
        ))
        return

    raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
