# clothing1m_data.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple
from torch.utils.data import Dataset
from PIL import Image



@dataclass
class Clothing1MItem:
    path: str
    noisy_label_idx: int
    verification_label: int  # 1=clean label says noisy was correct, 0=incorrect, -1=unknown

@dataclass
class Clothing1MSplits:
    classes: List[str]
    class_to_idx: Dict[str, int]
    noisy_only: List[Clothing1MItem]
    clean_all: List[Clothing1MItem]


# ----------------------------
# Datasets
# ----------------------------
class Clothing1MNoisyDataset(Dataset):
    def __init__(self, items: List[Clothing1MItem], tf):
        self.items = items
        self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(it.path).convert("RGB")
        return self.tf(img), it.noisy_label_idx

class Clothing1MCleanDataset(Dataset):
    """Returns (x, noisy_idx, verif, idx)."""
    def __init__(self, items: List[Clothing1MItem], tf):
        self.items = items
        self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(it.path).convert("RGB")
        return self.tf(img), it.noisy_label_idx, int(it.verification_label), i




def _read_pairs(path: str) -> List[Tuple[str, int]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            k, v = ln.split()
            out.append((k, int(v)))
    return out

def _read_keys(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def _load_classes(meta_dir: str) -> List[str]:
    names = os.path.join(meta_dir, "category_names_eng.txt")  # 14 classes
    if os.path.isfile(names):
        with open(names, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    # fallback: indices "0..13"
    return [str(i) for i in range(14)]

def build_clothing1m_splits(images_root: str, meta_dir: str) -> Clothing1MSplits:
    classes = _load_classes(meta_dir)
    class_to_idx = {c: i for i, c in enumerate(classes)}  # labels are already 0..13

    # KV maps: relative_image_path -> label_idx
    noisy_kv = dict(_read_pairs(os.path.join(meta_dir, "noisy_label_kv.txt")))
    clean_kv = dict(_read_pairs(os.path.join(meta_dir, "clean_label_kv.txt")))

    # Subset key lists
    noisy_keys = _read_keys(os.path.join(meta_dir, "noisy_train_key_list.txt"))
    clean_keys = []
    for nm in ["clean_train_key_list.txt", "clean_val_key_list.txt", "clean_test_key_list.txt"]:
        p = os.path.join(meta_dir, nm)
        if os.path.isfile(p): clean_keys += _read_keys(p)

    noisy_only, clean_all = [], []



    def _resolve_path(images_root: str, rel: str) -> str:
        # If metadata includes a leading "images/", normalize it away.
        rel_norm = rel[7:] if rel.startswith("images/") else rel

        # Prefer .../images if it exists; else use root
        base = os.path.join(images_root, "images")
        if not os.path.isdir(base):
            base = images_root
        return os.path.join(base, rel_norm)




    # Build clean_all with verification flags vs *noisy* label
    for rel in clean_keys:
        # dataset guarantees all labels retrievable from clean or noisy kv files
        noisy_y = noisy_kv.get(rel); clean_y = clean_kv.get(rel)
        if noisy_y is None or clean_y is None: 
            continue # if any missing, skip or mark unknown; here we skip rare inconsistencies
        verif = 1 if noisy_y == clean_y else 0
        full = _resolve_path(images_root, rel)
        clean_all.append(Clothing1MItem(full, noisy_y, verif))


    # Build noisy_only from the huge list
    for rel in noisy_keys:
        y = noisy_kv.get(rel)
        if y is None: 
            continue
        full = _resolve_path(images_root, rel)
        noisy_only.append(Clothing1MItem(full, y, -1))

    return Clothing1MSplits(classes, class_to_idx, noisy_only, clean_all)
