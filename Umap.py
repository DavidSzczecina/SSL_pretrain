import torch
import numpy as np
import umap
import matplotlib.pyplot as plt

from ssl_cifar_experiment import (
    ResNet18Small, get_dataloaders, extract_features, load_encoder_into_classifier
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Paths to your saved models
# -----------------------------
SUPERVISED_CKPT = "results_cifar100/baseline_cifar100_supE-10_noise-0.6_s-1/baseline_cifar100_noise0.60_freeze0.pt"
SSL_ENCODER_PATH = "pretrained_encoders/simclr_cifar100_e25_s1.pth"
SSL_FINETUNE_CKPT = "results_cifar100/simclr_cifar100_preE-25_supE-10_noise-0.6_s-1/ft_cifar100_noise0.60_freeze0.pt"

SUPERVISED_CKPT = "results_cifar100/baseline_cifar100_supE-10_noise-0.0_s-1/baseline_cifar100_noise0.00_freeze0.pt"
SSL_ENCODER_PATH = "pretrained_encoders/simclr_cifar100_e25_s1.pth"
SSL_FINETUNE_CKPT = "results_cifar100/simclr_cifar100_preE-25_supE-10_noise-0.0_s-1/ft_cifar100_noise0.00_freeze0.pt"


DATASET = "cifar100"
NOISE = 0.0
BATCH = 256


def load_supervised_model():
    model = ResNet18Small(num_classes=100).to(DEVICE)
    model.load_state_dict(torch.load(SUPERVISED_CKPT, map_location=DEVICE))
    return model


def load_ssl_finetuned_model():
    model = ResNet18Small(num_classes=100).to(DEVICE)
    load_encoder_into_classifier(SSL_ENCODER_PATH, model)
    model.load_state_dict(torch.load(SSL_FINETUNE_CKPT, map_location=DEVICE), strict=False)
    return model


def main():

    # --- Load data (test set for visualization) ---
    _, test_loader, test_ds, _ = get_dataloaders(
        dataset=DATASET,
        batch_size=BATCH,
        noise_rate=NOISE,
        workers=2,
        seed=1
    )

    # --- Load models ---
    sup = load_supervised_model()
    ssl = load_ssl_finetuned_model()

    # --- Extract embedding features ---
    print("Extracting supervised features...")
    X_sup, y = extract_features(sup, test_loader, DEVICE)

    print("Extracting SSL-pretrained+finetuned features...")
    X_ssl, _ = extract_features(ssl, test_loader, DEVICE)

    # --- Run UMAP ---
    reducer = umap.UMAP(n_components=1, random_state=1)

    Z_sup = reducer.fit_transform(X_sup)
    Z_ssl = reducer.fit_transform(X_ssl)

    # --- Plot ---
    fig, axs = plt.subplots(1, 2, figsize=(14,6))

    axs[0].scatter(Z_sup[:,0], Z_sup[:,1], c=y, cmap="tab10", s=5)
    axs[0].set_title("Supervised-only Features")

    axs[1].scatter(Z_ssl[:,0], Z_ssl[:,1], c=y, cmap="tab10", s=5)
    axs[1].set_title("SSL Pretrained → Finetuned Features")

    plt.tight_layout()
    plt.savefig("umap_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
