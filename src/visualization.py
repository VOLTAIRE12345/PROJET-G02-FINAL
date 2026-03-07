"""
visualization.py — Groupe G02
Génération de toutes les figures du rapport.
7 visualisations : courbes entraînement, heatmap régularisation, loss landscape 1D,
sharpness, convergence Optuna, matrice de confusion, généralisation gap.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi":     150,
})


# ─── 1. Courbes d'entraînement ───────────────────────────────────────────────
def plot_training_curves(history: dict, title: str = "Courbes d'entraînement",
                          save_name: str = "training_curves.png"):
    """Loss et Accuracy/F1 par époque pour une configuration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Validation")
    axes[0].set_title("Loss par époque")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-o",  label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   "r-o",  label="Val Acc")
    axes[1].plot(epochs, history["val_f1"],    "g--s", label="Val F1")
    axes[1].set_title("Accuracy & F1 par époque")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {path}")
    return path


# ─── 2. Heatmap Régularisation (Grid Search P02) ─────────────────────────────
def plot_regularization_heatmap(results_grid: dict,
                                 save_name: str = "regularization_heatmap.png"):
    """
    Heatmap Val F1 en fonction de (weight_decay × dropout_prob).
    results_grid : {(wd, dp): val_f1}
    Visualise les 12 combinaisons du Grid Search P02.
    """
    wds    = sorted(set(k[0] for k in results_grid))
    dps    = sorted(set(k[1] for k in results_grid))
    matrix = np.array([[results_grid.get((wd, dp), np.nan)
                        for dp in dps] for wd in wds])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                   vmin=np.nanmin(matrix) * 0.97, vmax=np.nanmax(matrix))
    ax.set_xticks(range(len(dps)))
    ax.set_yticks(range(len(wds)))
    ax.set_xticklabels([f"{d:.2f}" for d in dps])
    ax.set_yticklabels([f"{w:.0e}" for w in wds])
    ax.set_xlabel("Dropout probability")
    ax.set_ylabel("Weight decay")
    ax.set_title("Val F1-score : Weight Decay × Dropout\n"
                 "(Grid Search P02 — BERT-base / IMDb)")
    plt.colorbar(im, ax=ax, label="F1-score")

    for i in range(len(wds)):
        for j in range(len(dps)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {path}")
    return path


# ─── 3. Loss Landscape 1D ────────────────────────────────────────────────────
def plot_loss_landscape_1d(configs: list,
                            save_name: str = "loss_landscape_1d.png"):
    """
    Visualisation 1D du loss landscape pour plusieurs configurations.
    configs : liste de dict avec 'label', 'alphas', 'losses'
    Un minimum plat (courbe large) indique une meilleure généralisation.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    for cfg, color in zip(configs, colors):
        alphas = np.array(cfg["alphas"])
        losses = np.array(cfg["losses"])
        ax.plot(alphas, losses, label=cfg["label"], color=color, linewidth=2)
        idx_min = np.argmin(losses)
        ax.scatter(alphas[idx_min], losses[idx_min], color=color, s=80, zorder=5)

    ax.set_xlabel("Direction de perturbation α")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Landscape 1D — Comparaison des configurations\n"
                 "(Filter-wise normalization — Li et al., 2018)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {path}")
    return path


# ─── 4. Sharpness Comparison ─────────────────────────────────────────────────
def plot_sharpness_comparison(sharpness_dict: dict,
                               save_name: str = "sharpness_comparison.png"):
    """
    Bar chart de la sharpness par configuration.
    La barre bleue (plus basse) correspond au minimum le plus plat.
    sharpness_dict : {config_label: sharpness_value}
    """
    labels = list(sharpness_dict.keys())
    values = list(sharpness_dict.values())
    colors = ["#2196F3" if v == min(values) else "#FF5722" for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Configuration (weight_decay, dropout)")
    ax.set_ylabel("Sharpness (moyenne sur 5 directions)")
    ax.set_title("Sharpness des minima par configuration\n"
                 "(Bleu = minimum plat → meilleure généralisation)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {path}")
    return path


# ─── 5. Convergence Optuna ────────────────────────────────────────────────────
def plot_optuna_history(study, save_name: str = "optuna_history.png"):
    """
    Courbe de convergence Optuna + importance des hyperparamètres.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Convergence
    trial_numbers = [t.number for t in study.trials if t.value is not None]
    values        = [t.value  for t in study.trials if t.value is not None]
    best_so_far   = np.maximum.accumulate(values)
    axes[0].plot(trial_numbers, values,       "o", color="steelblue",
                 alpha=0.6, label="Trial", markersize=6)
    axes[0].plot(trial_numbers, best_so_far,  "-", color="crimson",
                 linewidth=2.5, label="Best so far")
    axes[0].set_xlabel("Numéro du trial")
    axes[0].set_ylabel("Val F1-score")
    axes[0].set_title("Convergence Optuna TPE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Importance des hyperparamètres
    try:
        importance = optuna.importance.get_param_importances(study)
        names = list(importance.keys())
        vals  = list(importance.values())
        axes[1].barh(names, vals, color="teal", edgecolor="black", linewidth=0.5)
        axes[1].set_xlabel("Importance relative")
        axes[1].set_title("Importance des hyperparamètres\n(FanovaImportanceEvaluator)")
        axes[1].grid(True, axis="x", alpha=0.3)
    except Exception as e:
        axes[1].text(0.5, 0.5, f"Importance non disponible\n({e})",
                     ha="center", va="center", transform=axes[1].transAxes)

    fig.suptitle("Optimisation Bayésienne — Optuna G02 P02", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {path}")
    return path


# ─── 6. Matrice de Confusion ──────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred,
                           class_names=("Négatif", "Positif"),
                           save_name: str = "confusion_matrix.png"):
    """Matrice de confusion du modèle sur le test set."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title("Matrice de Confusion — Test set\n(Meilleure configuration)")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {path}")
    return path


# ─── 7. Généralisation Gap ────────────────────────────────────────────────────
def plot_generalization_gap(configs_gap: dict,
                             save_name: str = "generalization_gap.png"):
    """
    Barres train vs val accuracy + courbe du gap par configuration.
    configs_gap : {label: {'train_acc': float, 'val_acc': float}}
    """
    labels     = list(configs_gap.keys())
    train_accs = [configs_gap[l]["train_acc"] for l in labels]
    val_accs   = [configs_gap[l]["val_acc"]   for l in labels]
    gaps       = [tr - va for tr, va in zip(train_accs, val_accs)]

    x     = np.arange(len(labels))
    width = 0.3
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, train_accs, width, label="Train Acc", color="#42A5F5", edgecolor="black")
    ax.bar(x + width/2, val_accs,   width, label="Val Acc",   color="#EF5350", edgecolor="black")

    ax2 = ax.twinx()
    ax2.plot(x, gaps, "ko--", linewidth=1.5, markersize=7, label="Généralisation gap")
    ax2.set_ylabel("Écart Train - Val (Généralisation gap)")
    ax2.axhline(0, color="gray", linestyle=":", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Écart train/val par configuration de régularisation\n"
                 "(Gap faible = meilleure généralisation)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {path}")
    return path


if __name__ == "__main__":
    # Test avec données synthétiques
    history_demo = {
        "train_loss": [0.65, 0.42, 0.30],
        "val_loss":   [0.60, 0.48, 0.45],
        "train_acc":  [0.63, 0.80, 0.88],
        "val_acc":    [0.68, 0.78, 0.80],
        "val_f1":     [0.67, 0.77, 0.79],
    }
    plot_training_curves(history_demo, title="Démo", save_name="demo_training.png")
    print("Visualisations de démo générées.")
