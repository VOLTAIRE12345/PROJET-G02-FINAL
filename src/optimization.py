"""
optimization.py — Groupe G02
Optimisation des hyperparamètres — Problématique P02 : Régularisation et Généralisation

MÉTHODE : Optuna (TPE Bayésien) — tel qu'assigné au groupe G02
PROTOCOLE P02 (conforme au sujet) :
  - Grid Search sur weight_decay  : [1e-5, 1e-4, 1e-3, 1e-2]
  - Grid Search sur dropout_prob  : [0.0, 0.1, 0.3]
  - Mesure : écart train/test, platitude des minima (sharpness)
  - Méthode globale : Optuna TPE pour trouver le meilleur HP global
"""
import os, sys, json, itertools
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_imdb_subset, get_dataloaders
from model_setup import get_model_and_tokenizer, build_optimizer
from train_eval  import train_model, evaluate


DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Données partagées entre trials (chargées une seule fois) ─────────────────
_SHARED_DATA = {}

def _get_shared_data():
    if not _SHARED_DATA:
        subsets = load_imdb_subset(
            num_train_per_class=300,
            num_val_per_class=100,
            num_test_per_class=150,
        )
        _SHARED_DATA["subsets"] = subsets
    return _SHARED_DATA["subsets"]


# ─── PARTIE 1 : Grid Search P02 (conforme protocole sujet) ────────────────────
# weight_decay × dropout — grille imposée par l'énoncé P02
WD_GRID      = [1e-5, 1e-4, 1e-3, 1e-2]
DROPOUT_GRID = [0.0, 0.1, 0.3]


def run_grid_search(subsets, lr=2e-5, num_epochs=3, batch_size=16,
                    warmup_ratio=0.1, max_length=200, verbose=True):
    """
    Grid Search exhaustif sur weight_decay × dropout_prob.
    Protocole P02 — 4 × 3 = 12 combinaisons.
    
    Returns:
        grid_results: dict {(wd, dp): {'val_f1', 'val_acc', 'train_acc', 'history'}}
    """
    grid_results = {}
    total = len(WD_GRID) * len(DROPOUT_GRID)
    print(f"\n{'='*60}")
    print(f"GRID SEARCH P02  |  {total} combinaisons  |  Device: {DEVICE}")
    print(f"{'='*60}")

    for i, (wd, dp) in enumerate(itertools.product(WD_GRID, DROPOUT_GRID), 1):
        label = f"WD={wd:.0e}  Drop={dp:.1f}"
        print(f"\n[{i:2d}/{total}] {label}")

        model, tokenizer = get_model_and_tokenizer(
            dropout_prob=dp, device=DEVICE
        )
        loaders = get_dataloaders(
            subsets, tokenizer, batch_size=batch_size, max_length=max_length
        )
        optimizer = build_optimizer(model, lr=lr, weight_decay=wd)
        history   = train_model(
            model, loaders, optimizer,
            num_epochs=num_epochs, warmup_ratio=warmup_ratio,
            device=DEVICE, patience=2, verbose=verbose
        )

        best_val_f1  = max(history["val_f1"])
        best_val_acc = max(history["val_acc"])
        best_tr_acc  = max(history["train_acc"])

        grid_results[(wd, dp)] = {
            "weight_decay":   wd,
            "dropout_prob":   dp,
            "val_f1":         best_val_f1,
            "val_acc":        best_val_acc,
            "train_acc":      best_tr_acc,
            "gen_gap":        best_tr_acc - best_val_acc,
            "history":        history,
            "model":          model,
        }

        # Sauvegarde JSON (sans modèle)
        record = {k: v for k, v in grid_results[(wd, dp)].items()
                  if k not in ("history", "model")}
        record["history"] = {k: v for k, v in history.items() if k != "train_time"}
        path = os.path.join(RESULTS_DIR,
                            f"grid_wd{wd:.0e}_dp{dp:.1f}.json".replace("-", ""))
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=float)

    # Résumé
    print(f"\n{'─'*60}")
    print("RÉSUMÉ GRID SEARCH :")
    best_key  = max(grid_results, key=lambda k: grid_results[k]["val_f1"])
    worst_key = min(grid_results, key=lambda k: grid_results[k]["val_f1"])
    print(f"  Meilleure config : WD={best_key[0]:.0e}  Drop={best_key[1]:.1f}"
          f"  →  Val F1 = {grid_results[best_key]['val_f1']:.4f}")
    print(f"  Pire config      : WD={worst_key[0]:.0e}  Drop={worst_key[1]:.1f}"
          f"  →  Val F1 = {grid_results[worst_key]['val_f1']:.4f}")

    # Sauvegarde résumé global
    summary = {
        "best_key": list(best_key),
        "best_val_f1": grid_results[best_key]["val_f1"],
        "all_results": [
            {"wd": k[0], "dp": k[1], "val_f1": v["val_f1"],
             "val_acc": v["val_acc"], "gen_gap": v["gen_gap"]}
            for k, v in grid_results.items()
        ]
    }
    with open(os.path.join(RESULTS_DIR, "grid_search_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    return grid_results


# ─── PARTIE 2 : Optuna TPE (méthode assignée au groupe G02) ───────────────────
# Espace de recherche étendu pour Optuna
SEARCH_SPACE = {
    "weight_decay": ("log_float", 1e-5, 1e-2),
    "dropout_prob": ("float",     0.0,  0.3),
    "lr":           ("log_float", 1e-6, 5e-4),
    "batch_size":   ("categorical", [8, 16]),
    "warmup_ratio": ("float",     0.0,  0.15),
    "num_epochs":   ("int",       2,    4),
}


def suggest_params(trial) -> dict:
    """Suggère un jeu d'hyperparamètres via Optuna."""
    params = {}
    for name, spec in SEARCH_SPACE.items():
        kind = spec[0]
        if kind == "log_float":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
    return params


def objective(trial) -> float:
    """Fonction objectif : maximise val_f1."""
    params  = suggest_params(trial)
    subsets = _get_shared_data()

    model, tokenizer = get_model_and_tokenizer(
        dropout_prob=params["dropout_prob"], device=DEVICE
    )
    loaders = get_dataloaders(
        subsets, tokenizer,
        batch_size=params["batch_size"],
        max_length=200,
    )
    optimizer = build_optimizer(
        model, lr=params["lr"], weight_decay=params["weight_decay"]
    )
    history = train_model(
        model, loaders, optimizer,
        num_epochs=params["num_epochs"],
        warmup_ratio=params["warmup_ratio"],
        device=DEVICE, patience=2, verbose=False,
    )

    val_f1 = max(history["val_f1"])
    trial.report(val_f1, step=len(history["val_f1"]))
    if trial.should_prune():
        raise optuna.TrialPruned()

    # Sauvegarde du trial
    result = {
        "trial_number": trial.number,
        "params": params,
        "val_f1": val_f1,
        "val_acc": max(history["val_acc"]),
        "train_acc": max(history["train_acc"]),
        "train_time": history.get("train_time", 0),
    }
    path = os.path.join(RESULTS_DIR, f"trial_{trial.number:03d}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)

    return val_f1


def run_optuna_study(n_trials=20, study_name="G02_P02_regularization",
                     storage=None) -> optuna.Study:
    """
    Lance l'étude Optuna TPE et retourne le study object.
    
    Args:
        n_trials:    nombre de trials (recommandé : 20 pour CPU)
        study_name:  nom de l'étude Optuna
        storage:     URL SQLite pour persistance optionnelle
    Returns:
        study Optuna avec les résultats
    """
    sampler = TPESampler(seed=42, n_startup_trials=5)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    print(f"\n{'='*60}")
    print(f"OPTUNA TPE — Étude : {study_name}")
    print(f"Trials : {n_trials}  |  Device : {DEVICE}")
    print(f"{'='*60}\n")

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(Exception,),
    )

    print(f"\n{'='*60}")
    print("MEILLEUR TRIAL OPTUNA")
    print(f"  Val F1     : {study.best_value:.4f}")
    print(f"  Paramètres :")
    for k, v in study.best_params.items():
        print(f"    {k:20s} = {v}")
    print(f"{'='*60}\n")

    # Sauvegarde résumé Optuna
    summary = {
        "best_value":  study.best_value,
        "best_params": study.best_params,
        "n_trials":    len(study.trials),
    }
    with open(os.path.join(RESULTS_DIR, "optuna_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    return study


if __name__ == "__main__":
    subsets = _get_shared_data()
    # Test rapide Grid Search (2 epochs seulement)
    results = run_grid_search(subsets, num_epochs=2, verbose=False)
    print("Grid search terminé.")
