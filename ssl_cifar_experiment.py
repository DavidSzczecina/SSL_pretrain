# main.py
from __future__ import annotations
import os, csv, time, argparse, random
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

from cleanlab.filter import find_label_issues

from ssl_pretrainers import get_pretrainer 

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

# ================ Data & Augs (from your current file) ================
CIFAR10_MEAN, CIFAR10_STD = (0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)
CIFAR100_MEAN, CIFAR100_STD = (0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761)

def get_cifar_norm(dataset: str):
    d = dataset.lower()
    if d == "cifar10":  return CIFAR10_MEAN, CIFAR10_STD, 10
    if d == "cifar100": return CIFAR100_MEAN, CIFAR100_STD, 100
    raise ValueError("dataset must be 'cifar10' or 'cifar100'")  

class SimCLRAugmentCIFAR:
    def __init__(self, size=32):
        from torchvision import transforms
        self.base = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8,0.8,0.8,0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
        ])
    def __call__(self, img):
        return self.base(img), self.base(img)

class NoisyLabelWrapper(Dataset):
    def __init__(self, base: Dataset, noise_rate: float, num_classes: int, seed: int = 42):
        import random as _r
        self.base = base; self.targets = []
        g = _r.Random(seed)
        for _, y in base:
            if g.random() < noise_rate:
                ny = g.randrange(num_classes)
                if ny == y: ny = (ny+1) % num_classes
                self.targets.append(ny)
            else:
                self.targets.append(y)
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, _ = self.base[i]
        return x, self.targets[i]

class NoisyLabelWrapperWithFlags(Dataset):
    def __init__(self, base: Dataset, noise_rate: float, num_classes: int, seed: int = 42):
        import random as _r
        self.base = base; self.noisy_targets=[]; self.is_corrupted=[]; self.original_targets=[]
        g = _r.Random(seed)
        for _, y in base:
            self.original_targets.append(y)
            if g.random() < noise_rate:
                ny = g.randrange(num_classes)
                if ny == y: ny = (ny+1) % num_classes
                self.noisy_targets.append(ny); self.is_corrupted.append(True)
            else:
                self.noisy_targets.append(y); self.is_corrupted.append(False)
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, _ = self.base[i]
        return x, self.noisy_targets[i]

def build_ssl_loader(dataset: str, batch_size: int, workers: int) -> DataLoader:
    tf_ssl = SimCLRAugmentCIFAR(32)
    root = os.path.join("data", dataset.lower())
    Base = datasets.CIFAR10 if dataset.lower()=="cifar10" else datasets.CIFAR100
    ds = Base(root=root, train=True, download=True, transform=tf_ssl)
    class _Wrap(Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            (xi,xj), _ = self.ds[idx]
            return (xi,xj), 0
    return DataLoader(_Wrap(ds), batch_size=batch_size, shuffle=True,
                      num_workers=workers, drop_last=True, pin_memory=True)  


class AsymmetricNoisyWrapper(Dataset):
    def __init__(self, base: Dataset, T: np.ndarray, seed: int=42):
        assert T.shape[0] == T.shape[1]
        self.base = base
        self.num_classes = T.shape[0]
        self.T = T
        rng = np.random.default_rng(seed)
        # pre-sample all noisy labels
        ys = []
        for _, y in base:
            ys.append(y)
        ys = np.array(ys)
        self.noisy_targets = ys.copy()
        for c in range(self.num_classes):
            idx = np.where(ys==c)[0]
            if len(idx)==0: continue
            self.noisy_targets[idx] = rng.choice(self.num_classes, size=len(idx), p=T[c])
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x,_ = self.base[i]
        return x, int(self.noisy_targets[i])




def get_dataloaders(dataset: str, batch_size: int, noise_rate: float,
                    workers: int, seed: int):
    mean, std, num_classes = get_cifar_norm(dataset)
    from torchvision import transforms
    sup_train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    root = os.path.join("data", dataset.lower())
    Base = datasets.CIFAR10 if dataset.lower()=="cifar10" else datasets.CIFAR100
    base_train = Base(root=root, train=True,  download=True, transform=sup_train_tf)
    test       = Base(root=root, train=False, download=True, transform=test_tf)
    train = NoisyLabelWrapper(base_train, noise_rate, num_classes, seed) if noise_rate>0 else base_train
    return (DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True),
            DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
            test, num_classes)

# ================ Model (your ResNet18Small) ================
class ResNet18Small(nn.Module):
    def __init__(self, imagenet_pretrain: bool = False, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        base = models.resnet18(weights=None)
        if imagenet_pretrain:
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x, return_feat: bool = False):
        x = self.features(x); x = torch.flatten(x, 1)
        if return_feat: return x
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
                     logger=print, save_path: Optional[str]=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    best = 0.0
    t0 = time.time()

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

        # ---- Save best model ----
        if save_path and test_acc > best:
            best = test_acc
            torch.save(model.state_dict(), save_path)

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
def predict_proba(model: nn.Module, loader: DataLoader, device: str, num_classes: int) -> np.ndarray:
    model.eval(); out=[]
    for x,_ in loader:
        x = x.to(device); logits = model(x)
        out.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(out, axis=0)

def cleanlab_on_corrupted_test(model: nn.Module, clean_test_ds: Dataset, device: str,
                               num_classes: int, noise_rate: float, seed: int,
                               out_dir: str, base_row: dict, metrics_path: str):
    wrapped = NoisyLabelWrapperWithFlags(clean_test_ds, noise_rate, num_classes, seed)
    loader = DataLoader(wrapped, batch_size=256, shuffle=False, num_workers=1, pin_memory=True)
    probs = predict_proba(model, loader, device, num_classes)
    noisy_labels = np.array(wrapped.noisy_targets, dtype=int)
    gt_corrupted = np.array(wrapped.is_corrupted, dtype=bool)

    issue_idx = find_label_issues(
        noisy_labels,
        pred_probs=probs,
        return_indices_ranked_by="self_confidence",
        filter_by="both",
        frac_noise= 1.0
    )
    pred_issue = np.zeros_like(gt_corrupted, dtype=bool); pred_issue[issue_idx]=True

    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score
    y_true = gt_corrupted.astype(int); y_pred = pred_issue.astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1]); TN,FP,FN,TP = cm.ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    #np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)

    row = dict(base_row)
    row.update({
        "cl_precision": prec, "cl_recall": rec, "cl_f1": f1, "cl_bal_acc": bal_acc,
        "cl_tp": int(TP), "cl_fp": int(FP), "cl_tn": int(TN), "cl_fn": int(FN),
        "cl_num_issues": int(pred_issue.sum()),
    })
    fields = ["mode","dataset","noise_rate","pretrain_name","pretrain_epochs","finetune_epochs","test_acc","timestamp",
              "cl_precision","cl_recall","cl_f1","cl_bal_acc","cl_tp","cl_fp","cl_tn","cl_fn","cl_num_issues"]
    ensure_dir(os.path.dirname(metrics_path))
    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header: w.writeheader()
        w.writerow(row)
    return {"cm": cm, "f1": f1, "bal_acc": bal_acc}


# CIFAR-N dataset with 90/10 split for label-error eval

class HumanNoisyWrapperSubset(Dataset):
    """
    A subset wrapper that returns (image, human_noisy_label) and keeps access
    to both noisy & clean labels for the selected indices.
    """
    def __init__(self, base: Dataset, human_labels: np.ndarray, clean_labels: np.ndarray, indices: np.ndarray):
        self.base = base
        self.human_labels = np.asarray(human_labels, dtype=int)
        self.clean_labels = np.asarray(clean_labels, dtype=int)
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        x, _ = self.base[j]
        return x, int(self.human_labels[j])

    def label_arrays_for_subset(self):
        idx = self.indices
        return self.human_labels[idx], self.clean_labels[idx]

def _cifarn_paths(root: str):
    f10 = os.path.join(root, "data/CIFAR-10_human.pt")
    f100 = os.path.join(root, "data/CIFAR-100_human.pt")
    return f10, f100

def get_dataloaders_cifarn(dataset: str, batch_size: int, workers: int,
                           seed: int, cifarn_root: str, c10n_label_type: str):
    use_c10n = dataset.lower() == "cifar-10n"
    use_c100n = dataset.lower() == "cifar-100n"
    assert use_c10n or use_c100n

    base_name = "cifar10" if use_c10n else "cifar100"
    mean, std, num_classes = get_cifar_norm(base_name)

    from torchvision import transforms, datasets as tvds
    sup_train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = eval_tf

    root = os.path.join("data", base_name)
    Base = tvds.CIFAR10 if base_name == "cifar10" else tvds.CIFAR100

    # Two views of the same train images: aug'd for training, deterministic for eval
    base_train      = Base(root=root, train=True,  download=True, transform=sup_train_tf)
    base_train_eval = Base(root=root, train=True,  download=True, transform=eval_tf)
    test            = Base(root=root, train=False, download=True, transform=test_tf)

    # Human label files
    f10, f100 = _cifarn_paths(cifarn_root)
    if use_c10n:
        data = torch.load(f10, map_location="cpu")
        clean = np.array(data["clean_label"])
        noisy = np.array(data[c10n_label_type])
        assert len(clean) == len(base_train), "CIFAR-10N length mismatch."
    else:
        data = torch.load(f100, map_location="cpu", weights_only=False)
        clean = np.array(data["clean_label"])
        noisy = np.array(data["noisy_label"])
        assert len(clean) == len(base_train), "CIFAR-100N length mismatch."

    # ----- 90/10 split (deterministic) -----
    N = len(base_train)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    split = int(0.9 * N)
    train_idx   = perm[:split]
    heldout_idx = perm[split:]

    # Datasets
    train_ds       = HumanNoisyWrapperSubset(base_train,      noisy, clean, train_idx)
    heldout_eval_ds= HumanNoisyWrapperSubset(base_train_eval, noisy, clean, heldout_idx)

    # Loaders
    train_loader        = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    heldout_eval_loader = DataLoader(heldout_eval_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader         = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # Label arrays aligned to the held-out subset only (for Cleanlab + metrics)
    human_held, clean_held = heldout_eval_ds.label_arrays_for_subset()

    # Return both loaders and the held-out labels
    return (train_loader, heldout_eval_loader, test_loader, test, num_classes, human_held, clean_held)

def cleanlab_on_cifarn_heldout(model: nn.Module,
                               heldout_eval_loader: DataLoader,
                               human_labels_heldout: np.ndarray,
                               clean_labels_heldout: np.ndarray,
                               device: str,
                               out_dir: str,
                               base_row: dict,
                               metrics_path: str):
    """
    Run label-error detection on the *held-out 10%*:
      - noisy_labels := human_labels_heldout
      - ground truth := (human_labels_heldout != clean_labels_heldout)
      - pred_probs    from model on held-out images (deterministic transforms)
    """
    # 1) Model probabilities on held-out images
    probs = []
    model.eval()
    for x, _ in heldout_eval_loader:
        x = x.to(device)
        logits = model(x)
        probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    pred_probs = np.concatenate(probs, axis=0)

    noisy_labels = human_labels_heldout.astype(int)
    gt_issue     = (human_labels_heldout != clean_labels_heldout).astype(bool)

    # 2) Cleanlab ranking
    issue_idx = find_label_issues(
        noisy_labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
        filter_by="both",
        frac_noise=1.0,
    )
    pred_issue = np.zeros_like(gt_issue, dtype=bool)
    pred_issue[issue_idx] = True

    # 3) Metrics
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score
    y_true = gt_issue.astype(int)
    y_pred = pred_issue.astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    row = dict(base_row)
    row.update({
        "cl_precision": float(prec),
        "cl_recall": float(rec),
        "cl_f1": float(f1),
        "cl_bal_acc": float(bal_acc),
        "cl_tp": int(TP), "cl_fp": int(FP), "cl_tn": int(TN), "cl_fn": int(FN),
        "cl_num_issues": int(pred_issue.sum()),
        # (optional) prevalence on held-out:
        "cl_true_issues": int(gt_issue.sum()),
        "cl_eval_split": "heldout10",
    })

    fields = [
        "mode","dataset","noise_rate","pretrain_name","pretrain_epochs","finetune_epochs","test_acc","timestamp",
        "cl_precision","cl_recall","cl_f1","cl_bal_acc","cl_tp","cl_fp","cl_tn","cl_fn","cl_num_issues",
        "cl_true_issues","cl_eval_split"
    ]
    ensure_dir(os.path.dirname(metrics_path))
    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header: w.writeheader()
        w.writerow(row)

    return {"cm": cm, "f1": f1, "bal_acc": bal_acc}




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
    p = argparse.ArgumentParser("CIFAR SSL → Train → Evaluate")
    # high-level mode
    p.add_argument("--mode", choices=["pretrain","train_eval"], default="train_eval")
    p.add_argument("--dataset", choices=["cifar10","cifar100","cifar-10n","cifar-100n"], default="cifar10")
    p.add_argument("--cifar10n-label-type", type=str,default="worse_label", choices=["worse_label","aggre_label"])
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
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--noise-rate", type=float, default=0.0)
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
    p.add_argument("--batch-size-ssl", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--proj-hidden", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.5)
    
    args = p.parse_args()


    set_seed(args.seed)
    device = args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu"


    use_c10n  = args.dataset.lower() == "cifar-10n"
    use_c100n = args.dataset.lower() == "cifar-100n"
    is_cifarn = use_c10n or use_c100n
    base_dataset = "cifar10" if (use_c10n or args.dataset.lower()=="cifar10") else "cifar100"

    # Keep results separated by user-facing dataset name (so 10N/100N get their own folders)
    results_root = args.results_root or f"results_{args.dataset}"
    ensure_dir(results_root); ensure_dir(args.pretrained_dir)

    # Experiment folder naming: if CIFAR-N, don't include a synthetic noise rate
    if not args.exp_name:
        if is_cifarn:
            # include the label source to be explicit
            label_src = args.cifar10n_label_type if use_c10n else "human"
            args.exp_name = f"{args.mode}_{args.dataset}_{label_src}"
        else:
            args.exp_name = f"{args.mode}_{args.dataset}_noise-{args.noise_rate:.2f}"


    results_root = args.results_root or f"results_{args.dataset}"
    ensure_dir(results_root); ensure_dir(args.pretrained_dir)

    # experiment folder
    if not args.exp_name:
        # clearer, includes mode + noise
        args.exp_name = f"{args.mode}_{args.dataset}_noise-{args.noise_rate:.2f}"
    exp_dir = os.path.join(results_root, args.exp_name); ensure_dir(exp_dir)
    log_path = os.path.join(exp_dir, "logs.txt")
    metrics_path = os.path.join(exp_dir, args.metrics_name)

    # data
    if is_cifarn:
        (train_loader, heldout_eval_loader, test_loader, clean_test_ds,
        num_classes, human_held, clean_held) = get_dataloaders_cifarn(
            dataset=args.dataset, batch_size=args.batch_size, workers=args.workers,
            seed=args.seed, cifarn_root="", c10n_label_type=args.cifar10n_label_type
        )
    else:
        train_loader, test_loader, clean_test_ds, num_classes = get_dataloaders(
            dataset=args.dataset, batch_size=args.batch_size,
            noise_rate=args.noise_rate, workers=args.workers, seed=args.seed
        )


    # -------- MODE: PRETRAIN (SSL only, then save encoder and exit) --------
    if args.mode == "pretrain":
        # build SSL data loader over FULL train set
        ssl_loader = build_ssl_loader(args.dataset, args.batch_size_ssl, args.workers)
        # backbone
        encoder = ResNet18Small(num_classes=num_classes, in_channels=3).to(device)
        # pretrainer
        pretrainer = get_pretrainer(args.pretrain_name)
        ssl_model = pretrainer.build(encoder, in_dim=512, proj_dim=args.proj_dim, hidden=args.proj_hidden)
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
    clf = ResNet18Small(args.imagenet_pretrain, num_classes=num_classes, in_channels=3).to(device)

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
    
    if is_cifarn:
        suffix = f"{args.dataset}"
        if use_c10n:
            suffix += f"_{args.cifar10n_label_type}"
        ckpt_name = ("ft_" if args.pretrained_encoder_path else "baseline_") \
                    + f"{suffix}_freeze{int(args.freeze_backbone)}.pt"
    else:
        ckpt_name = ("ft_" if args.pretrained_encoder_path else "baseline_") \
                    + f"{args.dataset}_noise{args.noise_rate:.2f}_freeze{int(args.freeze_backbone)}.pt"
    ckpt_path = os.path.join(exp_dir, ckpt_name)

    

    acc = train_supervised(clf, train_loader, test_loader, device,
                           epochs=args.epochs, lr=args.lr, wd=args.wd,
                           logger=lambda m: log_print(m, log_path), save_path=ckpt_path)
    log_print(f"[Result] Final Test Acc: {acc*100:.2f}%", log_path)

    # unified base row for metrics
    base_row = {
        "mode": "train_eval",
        "dataset": args.dataset,
        "noise_rate": args.noise_rate,
        "pretrain_name": args.pretrain_name if args.pretrained_encoder_path else "none",
        "pretrain_epochs": args.pretrain_epochs if args.pretrained_encoder_path else 0,
        "finetune_epochs": args.epochs,
        "test_acc": acc,
        "timestamp": time.time(),
    }

    # prediction + analysis: Cleanlab on corrupted TEST copy
    if is_cifarn:
        # Cleanlab on *held-out 10%* (never used for supervised training)
        cleanlab_on_cifarn_heldout(
            model=clf,
            heldout_eval_loader=heldout_eval_loader,
            human_labels_heldout=human_held,
            clean_labels_heldout=clean_held,
            device=device,
            out_dir=exp_dir,
            base_row=base_row,
            metrics_path=metrics_path
        )
    else:
        # Synthetic corruption path unchanged for non-CIFAR-N runs
        cleanlab_on_corrupted_test(
            model=clf, clean_test_ds=clean_test_ds, device=device,
            num_classes=num_classes, noise_rate=args.noise_rate, seed=args.seed,
            out_dir=exp_dir, base_row=base_row, metrics_path=metrics_path
        )
            

if __name__ == "__main__":
    main()
