# main.py
from __future__ import annotations
import os, csv, time, argparse, random
from typing import Optional
import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image

from cleanlab.filter import find_label_issues

from ssl_pretrainers import get_pretrainer 
from clothing1m_data import build_clothing1m_splits, Clothing1MNoisyDataset, Clothing1MCleanDataset

# ================ Utils ================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def log_print(msg: str, log_path: Optional[str] = None):
    print(msg, flush=True)
    if log_path:
        with open(log_path, "a") as f: f.write(msg + "\n")

# ================ Data & Augs ================


class SimCLRAugment224:
    def __init__(self):
        self.base = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.8,0.8,0.8,0.2)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)
            )
        ])

    def __call__(self, img):
        return self.base(img), self.base(img)


class CleanEvalWrapper(Dataset):
    """Wraps Clothing1MCleanDataset but returns only (x, y)."""
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, i):
        x, y, _, _ = self.base_ds[i]
        return x, y




def build_ssl_loader_C1M(images_dir, meta_dir,
                               batch_size, workers):

    splits = build_clothing1m_splits(images_dir, meta_dir)

    tf_ssl = SimCLRAugment224()

    class SSLDataset(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            img = Image.open(self.items[i].path).convert("RGB")
            return tf_ssl(img), 0

    ds = SSLDataset(splits.noisy_only)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=True,
        pin_memory=False
    )



def get_C1M_loaders(images_dir, meta_dir,
                           batch_size, workers):

    splits = build_clothing1m_splits(images_dir, meta_dir)

    num_classes = len(splits.classes)

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485,0.456,0.406),
            std=(0.229,0.224,0.225)
        ),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485,0.456,0.406),
            std=(0.229,0.224,0.225)
        ),
    ])

    ds_train = Clothing1MNoisyDataset(splits.noisy_only, train_tf)
    ds_clean = Clothing1MCleanDataset(splits.clean_all, eval_tf)

    train_loader = DataLoader(
        ds_train, batch_size=batch_size,
        shuffle=True, num_workers=workers,
        pin_memory=False, drop_last=True
    )

    # For accuracy evaluation
    ds_clean_eval = CleanEvalWrapper(ds_clean)

    clean_eval_loader = DataLoader(
        ds_clean_eval, batch_size=batch_size,
        shuffle=False, num_workers=workers
    )

    # For Cleanlab block
    clean_full_loader = DataLoader(
        ds_clean, batch_size=batch_size,
        shuffle=False, num_workers=workers
    )

    return train_loader, clean_eval_loader, clean_full_loader, num_classes




# ================ Model ================
class ResNet50(nn.Module):
    def __init__(self, imagenet_pretrain=True, num_classes=14):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if imagenet_pretrain else None
        base = models.resnet50(weights=weights)

        self.features = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x, return_feat=False):
        x = self.features(x)
        x = torch.flatten(x, 1)
        if return_feat:
            return x
        return self.fc(x)

# ================ Supervised train / eval ================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    ce = nn.CrossEntropyLoss()
    model.eval(); total=correct=0; loss_sum=0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x); loss = ce(logits,y)
        loss_sum += loss.item()*x.size(0)
        pred = logits.argmax(1); correct += (pred==y).sum().item(); total += x.size(0)
    return correct/total, loss_sum/total 

def train_supervised(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                     device: str, epochs: int, lr: float, wd: float,
                     logger=print, save_path: Optional[str]=None,
                     save_every: int = 10, is_finetune: bool=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    t0 = time.time()
    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_samples = 0

        # ---- Training loop ----
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            batch_size = x.size(0)
            total_train_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_train_loss / total_samples

        # ---- Evaluation ----
        train_noisy_acc = evaluate_train_noisy(model, train_loader, device)
        test_acc, test_loss = evaluate(model, test_loader, device)

        logger(
            f"[Supervised] Epoch {ep:03d} | "
            f"Train(noisy)Acc={train_noisy_acc*100:.2f}% | "
            f"TrainLoss={train_loss:.4f} | "
            f"Test(clean)Acc={test_acc*100:.2f}% | "
            f"TestLoss={test_loss:.4f}"
        )
        
        prefix = "ft_" if is_finetune else "baseline_"
        
        # ---- SAVE EVERY N EPOCHS ----        
        if save_path and ep % save_every == 0:
            save_ckpt = os.path.join(save_path, f"{prefix}C1M_epoch-{ep}.pt")
            torch.save(model.state_dict(), save_ckpt)
            logger(f"[Checkpoint] Saved {save_ckpt}")

        # ---- Save best model ----
        if save_path and test_acc > best:
            best = test_acc
            save_ckpt = os.path.join(save_path, f"{prefix}C1M_best.pt")
            torch.save(model.state_dict(), save_ckpt)

    logger(f"[Supervised] Done in {(time.time() - t0) / 60:.2f} min. Best Acc={best * 100:.2f}%")
    return best

def evaluate_train_noisy(model, loader, device):
    model.eval(); total=correct=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item(); total += x.size(0)
    return correct/total


@torch.no_grad()
def extract_features(encoder: nn.Module, loader: DataLoader, device: str):
    """
    Runs the encoder in frozen mode and returns (X, y) numpy arrays
    where X has shape [N, D] and y has shape [N].
    """
    encoder.eval()
    feats = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        z = encoder(x, return_feat=True)  # [B, D]
        feats.append(z.cpu().numpy())
        labels.append(y.numpy())
    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: str, num_classes: int) -> np.ndarray:
    model.eval(); out=[]
    for x,_ in loader:
        x = x.to(device); logits = model(x)
        out.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(out, axis=0)



# Metrics for label-error detection
def eval_label_issue_detection(pred_issue_idx: np.ndarray, verif: np.ndarray):
    """verif: 1=correct label, 0=incorrect (ground truth issue)."""
    gt = (verif == 0)
    pred = np.zeros_like(gt, dtype=bool)
    pred[pred_issue_idx] = True
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    tn = int(np.sum(~pred & ~gt))
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1)


# ================ Save/load encoder ================
def save_encoder(encoder: nn.Module, path: str):
    ensure_dir(os.path.dirname(path)); torch.save({"features": encoder.features.state_dict()}, path)

def load_encoder_into_classifier(path: str, clf: nn.Module):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("features", ckpt)
    missing, unexpected = clf.features.load_state_dict(state, strict=False)
    print(f"Loaded encoder from {path}.")
    #if (missing or unexpected) != None:
    #    print("Missing={missing}, Unexpected={unexpected}")

# ================ CLI ================
def main():
    p = argparse.ArgumentParser("C1M SSL → Train → Evaluate")
    # high-level mode
    p.add_argument("--mode", choices=["pretrain","train_eval","ssl_sklearn"], default="train_eval")
    p.add_argument("--dataset", type=str, choices=["C1M"], default="C1M")
    p.add_argument("--images_dir", type=str, default="")
    p.add_argument("--meta_dir", type=str, default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    
    # dirs
    p.add_argument("--exp-name", type=str, default="")
    p.add_argument("--results-root", type=str, default=None, help="defaults to results_<dataset>")
    p.add_argument("--pretrained-dir", type=str, default="pretrained_encoders")
    p.add_argument("--metrics-name", type=str, default="metrics.csv")
    # supervised
    p.add_argument("--imagenet-pretrain", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--pretrained-encoder-path", type=str, default="", help="optional: load this into classifier")
    # pretraining
    p.add_argument("--pretrain-name", type=str, default="simclr", help="which SSL method to run/use")
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--save-pretrained-encoder", type=str, default="", help="filename to save encoder during --mode pretrain")
    
    #simCLR
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--pretrain-wd", type=float, default=1e-4)
    p.add_argument("--batch-size-ssl", type=int, default=128)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--proj-hidden", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.5)
    
    args = p.parse_args()


    set_seed(args.seed)
    device = args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu"



    # Keep results separated by user-facing dataset name (so 10N/100N get their own folders)
    results_root = args.results_root or f"results_{args.dataset}"
    ensure_dir(results_root); ensure_dir(args.pretrained_dir)

    # Experiment folder naming
    if not args.exp_name:
        args.exp_name = f"{args.mode}_{args.dataset}"
    


    results_root = args.results_root or f"results_{args.dataset}"
    ensure_dir(results_root); ensure_dir(args.pretrained_dir)

    # experiment folder
    if not args.exp_name:
        # clearer, includes mode 
        args.exp_name = f"{args.mode}_{args.dataset}"
    exp_dir = os.path.join(results_root, args.exp_name); ensure_dir(exp_dir)
    log_path = os.path.join(exp_dir, "logs.txt")
    metrics_path = os.path.join(exp_dir, args.metrics_name)

    # data


    train_loader, clean_eval_loader, clean_full_loader, num_classes = get_C1M_loaders(
    args.images_dir, args.meta_dir, args.batch_size, args.workers)


    # -------- MODE: PRETRAIN (SSL only, then save encoder and exit) --------
    if args.mode == "pretrain":
        # build SSL data loader over FULL train set
        ssl_loader = build_ssl_loader_C1M(args.images_dir, args.meta_dir, args.batch_size_ssl, args.workers)
        # backbone
        encoder = ResNet50(imagenet_pretrain=args.imagenet_pretrain, num_classes=num_classes).to(device)
        # pretrainer
        pretrainer = get_pretrainer(args.pretrain_name)
        ssl_model = pretrainer.build(encoder, in_dim = 2048, proj_dim=args.proj_dim, hidden=args.proj_hidden)
        # fit

        def _save_fn(ssl_model, ep):
            enc = pretrainer.extract_encoder(ssl_model)
            path = os.path.join(args.pretrained_dir, f"{args.pretrain_name}_{args.dataset}_e{ep}_s{args.seed}.pth")
            save_encoder(enc, path)
            log_print(f"[Pretrain] Saved encoder @e{ep} -> {path}", log_path)

        pretrainer.fit(
            ssl_model, ssl_loader, device,
            epochs=args.pretrain_epochs, lr=args.pretrain_lr, wd=args.pretrain_wd,
            temperature=args.temperature,
            save_every=args.save_every, save_fn=_save_fn, logger=lambda m: log_print(m, log_path)
        )

        # save encoder if asked
        if args.save_pretrained_encoder:
            path = os.path.join(args.pretrained_dir, args.save_pretrained_encoder)
            save_encoder(pretrainer.extract_encoder(ssl_model), path)
            log_print(f"[Pretrain] Saved encoder -> {path}", log_path)
        return

    # -------- MODE: TRAIN + EVAL --------
    # classifier
    clf = ResNet50(imagenet_pretrain=args.imagenet_pretrain, num_classes=num_classes).to(device)
    # optional load of a *previously* pretrained encoder

    encoder_path = os.path.join(args.pretrained_dir, args.pretrained_encoder_path)
    if args.pretrained_encoder_path and os.path.exists(encoder_path):
        load_encoder_into_classifier(encoder_path, clf)
        print("Pretrained encoder being used")
        if args.freeze_backbone:
            for p in clf.features.parameters(): p.requires_grad = False
            print("Freezing backbone")
    else:
        print("No pretrained encoder used")
    
 


    acc = train_supervised(clf, train_loader, clean_eval_loader, device,
                           epochs=args.epochs, lr=args.lr, wd=args.wd,
                           logger=lambda m: log_print(m, log_path), save_path=exp_dir, 
                           save_every=5, is_finetune=bool(args.pretrained_encoder_path))

    log_print(f"[Result] Final Test Acc: {acc*100:.2f}%", log_path)

    # unified base row for metrics
    base_row = {
        "mode": "train_eval",
        "dataset": args.dataset,
        "pretrain_name": args.pretrain_name if args.pretrained_encoder_path else "none",
        "pretrain_epochs": args.pretrain_epochs if args.pretrained_encoder_path else 0,
        "finetune_epochs": args.epochs,
        "test_acc": acc,
        "timestamp": time.time(),
    }


    # Predict on clean_all
    # Reconstruct splits to access verification labels
    splits = build_clothing1m_splits(args.images_dir, args.meta_dir)

    # Collect probabilities on clean set
    clf.eval()
    probs = []
    noisy_labels = []
    verification = []

    with torch.no_grad():
        for x, y, verif, idx in clean_full_loader:
            x = x.to(device)
            logits = clf(x)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)

            noisy_labels.append(y.numpy())
            verification.append(verif.numpy())

    probs = np.concatenate(probs, axis=0)
    noisy_labels = np.concatenate(noisy_labels, axis=0)
    verification = np.concatenate(verification, axis=0)

    # Save probabilities
    np.save(os.path.join(exp_dir, "pred_probs_clean_all.npy"), probs)

    # Run Cleanlab
    issue_idx = find_label_issues(
        labels=noisy_labels,
        pred_probs=probs,
        return_indices_ranked_by="self_confidence"
    )

    np.save(os.path.join(exp_dir, "issue_indices.npy"), issue_idx)

    # Evaluate against ground-truth verification labels
    metrics = eval_label_issue_detection(issue_idx, verification)

    print("\nCleanlab Detection Metrics:")
    print(json.dumps(metrics, indent=2))

    with open(os.path.join(exp_dir, "cleanlab_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
            

if __name__ == "__main__":
    main()
