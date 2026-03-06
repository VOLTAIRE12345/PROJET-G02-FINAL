"""
main_experiment.py — Groupe G02
Script principal : pipeline expérimental complet.

GROUPE  : G02
DATASET : IMDb (D01) — Sentiment Analysis
MODÈLE  : BERT-base-uncased (M02)
PROBLÉMATIQUE : P02 — Régularisation et Généralisation
MÉTHODE : Optuna (TPE Bayésien)
MÉTRIQUE : F1-score (binaire)

PIPELINE :
  Étape 1 : Chargement et sous-échantillonnage des données IMDb
  Étape 2 : Grid Search P02 — weight_decay × dropout (protocole imposé)
  Étape 3 : Optimisation globale Optuna TPE (20 trials)
  Étape 4 : Entraînement des configurations clés (5 configs)
  Étape 5 : Analyse Loss Landscape + Sharpness
  Étape 6 : Génération des figures
  Étape 7 : Évaluation finale sur test set + résumé

Durée estimée GPU T4 : ~25-35 min
Durée estimée CPU    : ~3-5h (réduire n_trials et num_epochs si nécessaire)
"""
import os, sys, json, copy
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader    import load_imdb_subset, get_dataloaders
from src.model_setup    import get_model_and_tokenizer, build_optimizer
from src.train_eval     import train_model, evaluate
from src.optimization   import run_grid_search, run_optuna_study, _get_shared_data
from src.loss_landscape import analyze_configs
from src.visualization  import (
    plot_training_curves, plot_regularization_heatmap,
    plot_loss_landscape_1d, plot_sharpness_comparison,
    plot_optuna_history, plot_confusion_matrix,
    plot_generalization_gap
)

# ─── Configuration globale ────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Hyperparamètres fixes
NUM_EPOCHS    = 3      # réduire à 2 si CPU lent
BATCH_SIZE    = 16
WARMUP_RATIO  = 0.1
MAX_LENGTH    = 200    # tronqué pour économiser la mémoire
LR_DEFAULT    = 2e-5
N_OPTUNA      = 20     # nombre de trials Optuna (réduire à 10 si CPU)

print(f"\n{'='*65}")
print("  PROJET G02 — Fine-tuning BERT-base / IMDb — P02 Régularisation")
print(f"  Device : {DEVICE}")
print(f"  GPU disponible : {'Oui' if torch.cuda.is_available() else 'Non'}")
print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 : Chargement des données
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("ÉTAPE 1 : Chargement et sous-échantillonnage des données")
print("═"*65)

subsets = load_imdb_subset(
    num_train_per_class=300,   # 600 exemples train au total
    num_val_per_class=100,     # 200 exemples validation
    num_test_per_class=150,    # 300 exemples test
)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 : Grid Search P02 (protocole imposé par le sujet)
# weight_decay ∈ {1e-5, 1e-4, 1e-3, 1e-2}  ×  dropout ∈ {0.0, 0.1, 0.3}
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("ÉTAPE 2 : Grid Search P02 (12 combinaisons WD × Dropout)")
print("═"*65)

grid_results = run_grid_search(
    subsets,
    lr=LR_DEFAULT,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    warmup_ratio=WARMUP_RATIO,
    max_length=MAX_LENGTH,
    verbose=True,
)

# Heatmap Grid Search (immédiatement disponible)
grid_f1 = {k: v["val_f1"] for k, v in grid_results.items()}
plot_regularization_heatmap(grid_f1)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 : Optimisation Optuna TPE (méthode assignée G02)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("ÉTAPE 3 : Optimisation bayésienne Optuna TPE")
print("═"*65)

# Injecter les données partagées pour les trials Optuna
from src.optimization import _SHARED_DATA
_SHARED_DATA["subsets"] = subsets

study      = run_optuna_study(n_trials=N_OPTUNA)
best_params = study.best_params

print(f"\nMeilleurs HP Optuna : {best_params}")
plot_optuna_history(study)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 : Entraînement des 5 configurations clés
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("ÉTAPE 4 : Entraînement des 5 configurations de référence")
print("═"*65)

# Sélectionner la meilleure et la pire config du Grid Search
best_grid_key  = max(grid_results, key=lambda k: grid_results[k]["val_f1"])
worst_grid_key = min(grid_results, key=lambda k: grid_results[k]["val_f1"])

CONFIGS_TO_COMPARE = {
    "Défaut (WD=0, Drop=0.1)": {
        "weight_decay": 0.0,
        "dropout_prob": 0.1,
        "lr":           LR_DEFAULT,
    },
    f"Grid Best (WD={best_grid_key[0]:.0e}, Drop={best_grid_key[1]:.1f})": {
        "weight_decay": best_grid_key[0],
        "dropout_prob": best_grid_key[1],
        "lr":           LR_DEFAULT,
    },
    f"Grid Worst (WD={worst_grid_key[0]:.0e}, Drop={worst_grid_key[1]:.1f})": {
        "weight_decay": worst_grid_key[0],
        "dropout_prob": worst_grid_key[1],
        "lr":           LR_DEFAULT,
    },
    "Fort WD (WD=1e-2, Drop=0)": {
        "weight_decay": 1e-2,
        "dropout_prob": 0.0,
        "lr":           LR_DEFAULT,
    },
    "Optuna Best": {
        "weight_decay": best_params.get("weight_decay", 1e-4),
        "dropout_prob": best_params.get("dropout_prob", 0.1),
        "lr":           best_params.get("lr", LR_DEFAULT),
    },
}

trained_configs  = []
histories_all    = {}
tokenizer_shared = None

for cfg_name, hp in CONFIGS_TO_COMPARE.items():
    print(f"\n  ── Configuration : {cfg_name} ──")
    model, tokenizer = get_model_and_tokenizer(
        dropout_prob=hp["dropout_prob"], device=DEVICE
    )
    if tokenizer_shared is None:
        tokenizer_shared = tokenizer

    loaders   = get_dataloaders(subsets, tokenizer, batch_size=BATCH_SIZE,
                                 max_length=MAX_LENGTH)
    optimizer = build_optimizer(model, lr=hp["lr"],
                                 weight_decay=hp["weight_decay"])
    history   = train_model(model, loaders, optimizer,
                             num_epochs=NUM_EPOCHS, warmup_ratio=WARMUP_RATIO,
                             device=DEVICE, patience=2, verbose=True)

    histories_all[cfg_name] = history
    trained_configs.append({
        "label":   cfg_name,
        "model":   model,
        "history": history,
        "hp":      hp,
    })

    # Courbe d'entraînement individuelle
    safe_name = cfg_name.replace(" ", "_").replace("/", "_").replace("=", "").replace(",", "")
    plot_training_curves(
        history,
        title=f"Entraînement — {cfg_name}",
        save_name=f"training_{safe_name}.png",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 : Analyse Loss Landscape & Sharpness
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("ÉTAPE 5 : Analyse Loss Landscape (1D) et Sharpness")
print("═"*65)

loaders_shared = get_dataloaders(subsets, tokenizer_shared,
                                  batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
landscape_results = analyze_configs(trained_configs, loaders_shared, DEVICE)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 : Génération des figures de synthèse
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("ÉTAPE 6 : Génération des figures")
print("═"*65)

# Loss landscape 1D (toutes configs)
landscape_plot_data = [
    {
        "label":  label,
        "alphas": np.array(res["alphas"]),
        "losses": np.array(res["losses"]),
    }
    for label, res in landscape_results.items()
]
plot_loss_landscape_1d(landscape_plot_data)

# Sharpness comparison
sharpness_dict = {label: res["sharpness"] for label, res in landscape_results.items()}
plot_sharpness_comparison(sharpness_dict)

# Généralisation gap
gap_dict = {
    label: {"train_acc": res["train_acc"], "val_acc": res["val_acc"]}
    for label, res in landscape_results.items()
}
plot_generalization_gap(gap_dict)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 7 : Évaluation finale sur test set
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("ÉTAPE 7 : Évaluation finale sur le test set")
print("═"*65)

# Sélectionner le meilleur modèle selon val F1
best_cfg = max(trained_configs, key=lambda c: max(c["history"]["val_f1"]))
print(f"  Meilleure configuration (val F1) : {best_cfg['label']}")

final_model = best_cfg["model"]
test_metrics = evaluate(final_model, loaders_shared["test"], DEVICE)
print(f"\n  ── Résultats sur Test Set ──")
print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
print(f"  F1-score : {test_metrics['f1']:.4f}")
print(f"  Loss     : {test_metrics['loss']:.4f}")

# Matrice de confusion
final_model.eval()
all_preds, all_labels_list = [], []
with torch.no_grad():
    for batch in loaders_shared["test"]:
        batch   = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = final_model(**batch)
        preds   = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels_list.extend(batch["labels"].cpu().numpy())

plot_confusion_matrix(all_labels_list, all_preds)

# Sauvegarde résumé final JSON
final_summary = {
    "groupe":           "G02",
    "dataset":          "IMDb (D01)",
    "modele":           "BERT-base-uncased (M02)",
    "problematique":    "P02 — Régularisation et Généralisation",
    "methode":          "Optuna TPE",
    "best_config":      best_cfg["label"],
    "best_hp":          best_cfg["hp"],
    "test_metrics":     test_metrics,
    "sharpness":        sharpness_dict,
    "generalization_gaps": {k: v["train_acc"] - v["val_acc"]
                            for k, v in gap_dict.items()},
    "optuna_best_params": best_params,
    "optuna_best_val_f1": study.best_value,
    "grid_best_key":    list(best_grid_key),
    "grid_best_val_f1": grid_results[best_grid_key]["val_f1"],
}
with open(os.path.join(RESULTS_DIR, "final_summary.json"), "w") as f:
    json.dump(final_summary, f, indent=2, default=float)

# Résumé final dans la console
print(f"\n{'='*65}")
print("  EXPÉRIENCE TERMINÉE — RÉSUMÉ")
print(f"{'='*65}")
print(f"  Meilleure config  : {best_cfg['label']}")
print(f"  Test Accuracy     : {test_metrics['accuracy']:.4f}")
print(f"  Test F1-score     : {test_metrics['f1']:.4f}")
print(f"  Config Sharpness  : {sharpness_dict.get(best_cfg['label'], 'N/A'):.5f}")
print(f"  Figures générées  : {FIGURES_DIR}/")
print(f"  Résultats JSON    : {RESULTS_DIR}/")
print(f"{'='*65}\n")
