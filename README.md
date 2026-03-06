# Projet G02 — Fine-tuning BERT sur IMDb
## P02 : Régularisation et Généralisation

> **Cours** : ML & Optimisation | **Date limite** : 13 mars 2026  
> **Enseignant** : MBIA NDI Marie Thérèse — mbialaura12@gmail.com

---

## Informations du groupe

| Champ | Valeur |
|---|---|
| Groupe | G02 |
| Dataset | IMDb Reviews (D01) — Analyse de sentiment |
| Modèle | BERT-base-uncased (M02, 110M paramètres) |
| Problématique | P02 — Régularisation et Généralisation |
| Méthode d'optimisation | Optuna (TPE Bayésien) |
| Métrique principale | F1-score (binaire) |

---

## Problématique

**Question centrale :** Comment le `weight_decay` et le `dropout` affectent-ils la généralisation de BERT lors du fine-tuning sur IMDb ?

**Protocole P02 (conforme au sujet) :**
- Grid Search sur `weight_decay` ∈ {1e-5, 1e-4, 1e-3, 1e-2}
- Grid Search sur `dropout_prob` ∈ {0.0, 0.1, 0.3}
- Mesure de l'écart train/validation (généralisation gap)
- Analyse de la platitude des minima (sharpness)

---

## Structure du projet

```
G02_Final/
├── README.md                  ← Ce fichier
├── requirements.txt           ← Dépendances Python
├── main_experiment.py         ← Script principal (pipeline complet)
│
├── src/
│   ├── data_loader.py         ← Chargement et sous-échantillonnage IMDb
│   ├── model_setup.py         ← BERT + AdamW avec weight decay découplé
│   ├── train_eval.py          ← Entraînement, évaluation, early stopping
│   ├── optimization.py        ← Grid Search P02 + Optuna TPE
│   ├── loss_landscape.py      ← Landscape 1D + Sharpness (Li et al., 2018)
│   └── visualization.py       ← 7 figures pour le rapport
│
├── notebooks/
│   └── G02_BERT_IMDb_Colab.ipynb  ← Notebook exécutable sur Google Colab
│
├── results/                   ← JSON des résultats (généré à l'exécution)
└── figures/                   ← Figures PNG (générées à l'exécution)
```

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/VOLTAIRE12345/Projet_ML_Opt_G02.git
cd Projet_ML_Opt_G02

# Installer les dépendances
pip install -r requirements.txt
```

---

## Exécution

### Option 1 — Google Colab (recommandé)

1. Ouvrir `notebooks/G02_BERT_IMDb_Colab.ipynb` dans [Google Colab](https://colab.research.google.com)
2. `Exécution > Modifier le type d'exécution > GPU T4`
3. `Ctrl+F9` pour exécuter toutes les cellules
4. Durée estimée : ~25-35 min sur GPU T4

### Option 2 — Script Python local

```bash
# Lancer le pipeline complet
python main_experiment.py
```

**Pour CPU avec ressources limitées**, réduire dans `main_experiment.py` :
```python
N_OPTUNA   = 10    # au lieu de 20
NUM_EPOCHS = 2     # au lieu de 3
```

---

## Choix techniques

### Weight Decay découplé (pratique HuggingFace)

```python
no_decay = ["bias", "LayerNorm.weight"]
grouped_params = [
    {"params": [...],  "weight_decay": weight_decay},   # poids régularisés
    {"params": [...],  "weight_decay": 0.0},             # biais + LN exclus
]
```

Le weight decay n'est **pas** appliqué aux biais et aux poids LayerNorm, conformément à la recommandation de Devlin et al. (2019) et à la pratique standard HuggingFace.

### Sharpness robuste (5 directions)

```python
sharpness = mean over 5 random directions of |L(θ + ε·d) - L(θ)|
```

La moyenne sur 5 directions aléatoires (plutôt qu'une seule) rend la mesure plus stable et reproductible.

### Filter-wise Normalization (Li et al., 2018)

La direction de perturbation pour le loss landscape est normalisée par la norme du paramètre correspondant, ce qui rend la comparaison entre configurations équitable.

---

## Résultats obtenus

*(Mis à jour après exécution)*

| Configuration | Val F1 | Sharpness | Gen. Gap |
|---|---|---|---|
| Défaut (WD=0, Drop=0.1) | — | — | — |
| Grid Best | — | — | — |
| Grid Worst | — | — | — |
| Fort WD (1e-2) | — | — | — |
| Optuna Best | — | — | — |

**Test set (meilleure config) :**
- Accuracy : —
- F1-score : —

---

## Figures générées

| Figure | Description |
|---|---|
| `regularization_heatmap.png` | Heatmap Val F1 : WD × Dropout (Grid Search P02) |
| `loss_landscape_1d.png` | Loss landscape 1D pour les 5 configurations |
| `sharpness_comparison.png` | Sharpness par configuration |
| `generalization_gap.png` | Écart train/val par configuration |
| `optuna_history.png` | Convergence Optuna + importance HP |
| `confusion_matrix.png` | Matrice de confusion sur test set |
| `training_*.png` | Courbes d'entraînement par configuration |

---

## Références

- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.
- Li et al. (2018). *Visualizing the Loss Landscape of Neural Nets.* NeurIPS.
- Keskar et al. (2017). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima.* ICLR.
- Akiba et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD.
