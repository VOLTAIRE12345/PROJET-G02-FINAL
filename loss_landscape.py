"""
loss_landscape.py — Groupe G02
Analyse du loss landscape — méthode 1D simplifiée, compatible CPU.

Références :
  - Li et al. (2018) "Visualizing the Loss Landscape of Neural Nets"
    → Filter-wise normalization pour comparer les configs équitablement
  - Keskar et al. (2017) "On Large-Batch Training for Deep Learning"
    → Définition de la sharpness
"""
import os, copy
import numpy as np
import torch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Évaluation rapide sur sous-ensemble ──────────────────────────────────────
def evaluate_on_subset(model, loader, device, n_samples=64):
    """Calcule la loss moyenne sur n_samples exemples du loader."""
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            bs = batch["labels"].size(0)
            total_loss += outputs.loss.item() * bs
            n += bs
            if n >= n_samples:
                break
    return total_loss / max(n, 1)


# ─── Loss Landscape 1D ────────────────────────────────────────────────────────
def compute_loss_landscape_1d(model, loader, device, n_points=15, epsilon=0.05):
    """
    Perturbation 1D du modèle autour du point courant θ.
    Utilise la filter-wise normalization de Li et al. (2018) :
    chaque direction est normalisée par la norme du paramètre correspondant,
    ce qui rend la comparaison entre configurations équitable.
    
    Args:
        model:    modèle PyTorch (après entraînement)
        loader:   DataLoader de validation
        device:   device PyTorch
        n_points: nombre de points sur l'axe α
        epsilon:  amplitude de perturbation (±epsilon)
    Returns:
        (alphas, losses) : listes de floats
    """
    model.eval()
    original_params = [p.clone().detach() for p in model.parameters()]

    # Direction aléatoire avec filter-wise normalization
    direction = []
    for p in model.parameters():
        d      = torch.randn_like(p)
        norm_p = p.data.norm() + 1e-8   # norme du paramètre
        d      = d / d.norm() * norm_p  # normalisation par norme paramètre
        direction.append(d)

    alphas = np.linspace(-epsilon, epsilon, n_points)
    losses = []

    for alpha in alphas:
        for p, p0, d in zip(model.parameters(), original_params, direction):
            p.data = p0 + alpha * d
        loss = evaluate_on_subset(model, loader, device, n_samples=64)
        losses.append(loss)

    # Restauration du modèle original
    for p, p0 in zip(model.parameters(), original_params):
        p.data = p0.clone()

    return alphas.tolist(), losses


# ─── Sharpness ────────────────────────────────────────────────────────────────
def compute_sharpness(model, loader, device, rho=0.05, n_directions=5):
    """
    Sharpness = moyenne sur n_directions de |L(θ + ε·d) - L(θ)|
    où ||ε·d|| = rho.
    
    La moyenne sur plusieurs directions (au lieu d'une seule) rend la mesure
    plus robuste et moins sensible à l'aléatoire (Keskar et al., 2017).
    
    Args:
        model:        modèle PyTorch
        loader:       DataLoader de validation
        device:       device PyTorch
        rho:          rayon de perturbation
        n_directions: nombre de directions aléatoires
    Returns:
        sharpness (float)
    """
    base_loss       = evaluate_on_subset(model, loader, device, n_samples=128)
    original_params = [p.clone().detach() for p in model.parameters()]
    max_deltas      = []

    for _ in range(n_directions):
        direction   = [torch.randn_like(p) for p in model.parameters()]
        total_norm  = sum(d.norm()**2 for d in direction).sqrt() + 1e-8
        direction   = [rho * d / total_norm for d in direction]  # ||ε|| = rho

        for p, p0, d in zip(model.parameters(), original_params, direction):
            p.data = p0 + d

        perturbed_loss = evaluate_on_subset(model, loader, device, n_samples=128)
        max_deltas.append(abs(perturbed_loss - base_loss))

        for p, p0 in zip(model.parameters(), original_params):
            p.data = p0.clone()

    sharpness = float(np.mean(max_deltas))
    return sharpness


# ─── Analyse complète d'une liste de configurations ──────────────────────────
def analyze_configs(configs: list, loaders: dict, device):
    """
    Calcule landscape + sharpness pour chaque configuration entraînée.
    
    Args:
        configs: liste de dict avec clés 'label', 'model', 'history'
        loaders: dict avec clé 'validation'
        device:  device PyTorch
    Returns:
        dict {label: {alphas, losses, sharpness, val_f1, val_acc, train_acc, gen_gap}}
    """
    results = {}
    for cfg in configs:
        label = cfg["label"]
        model = cfg["model"]
        print(f"\nAnalyse landscape : {label}")

        alphas, losses = compute_loss_landscape_1d(
            model, loaders["validation"], device, n_points=15, epsilon=0.05
        )
        sharpness = compute_sharpness(
            model, loaders["validation"], device, rho=0.05, n_directions=5
        )

        best_val_f1  = max(cfg["history"]["val_f1"])
        best_val_acc = max(cfg["history"]["val_acc"])
        best_tr_acc  = max(cfg["history"]["train_acc"])

        results[label] = {
            "alphas":             alphas,
            "losses":             losses,
            "sharpness":          sharpness,
            "val_f1":             best_val_f1,
            "val_acc":            best_val_acc,
            "train_acc":          best_tr_acc,
            "generalization_gap": best_tr_acc - best_val_acc,
        }
        print(f"  Sharpness       = {sharpness:.5f}")
        print(f"  Val F1          = {best_val_f1:.4f}")
        print(f"  Gen. gap        = {best_tr_acc - best_val_acc:.4f}")

    return results


if __name__ == "__main__":
    print("Module loss_landscape chargé avec succès.")
