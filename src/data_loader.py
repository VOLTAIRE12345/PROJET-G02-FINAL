"""
data_loader.py — Groupe G02
Dataset : IMDb (D01) | Modèle : BERT-base (M02) | Problématique : P02 | Méthode : Optuna
Sous-échantillonnage équilibré adapté aux contraintes CPU/mémoire.
"""
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


def load_imdb_subset(num_train_per_class=300, num_val_per_class=100,
                     num_test_per_class=150, seed=42):
    """
    Charge le dataset IMDb et retourne un sous-ensemble équilibré.
    Adapté aux contraintes matérielles (CPU, RAM limitée).
    
    Args:
        num_train_per_class: nb d'exemples par classe pour l'entraînement
        num_val_per_class:   nb d'exemples par classe pour la validation
        num_test_per_class:  nb d'exemples par classe pour le test
        seed: graine aléatoire pour la reproductibilité
    Returns:
        dict avec clés 'train', 'validation', 'test'
    """
    print("Chargement du dataset IMDb...")
    raw = load_dataset("imdb")
    rng = np.random.default_rng(seed)

    def _sample(split_data, n_per_class):
        pos = [ex for ex in split_data if ex["label"] == 1]
        neg = [ex for ex in split_data if ex["label"] == 0]
        n = min(n_per_class, len(pos), len(neg))
        idx_pos = rng.choice(len(pos), n, replace=False).tolist()
        idx_neg = rng.choice(len(neg), n, replace=False).tolist()
        subset = [pos[i] for i in idx_pos] + [neg[i] for i in idx_neg]
        rng.shuffle(subset)
        return subset

    # Diviser le train en train + validation (85% / 15%)
    train_full = list(raw["train"])
    rng.shuffle(train_full)
    pivot = int(0.85 * len(train_full))
    raw_train, raw_val_pool = train_full[:pivot], train_full[pivot:]

    subsets = {
        "train":      _sample(raw_train,        num_train_per_class),
        "validation": _sample(raw_val_pool,      num_val_per_class),
        "test":       _sample(list(raw["test"]), num_test_per_class),
    }

    for split, data in subsets.items():
        n_pos = sum(1 for d in data if d["label"] == 1)
        n_neg = sum(1 for d in data if d["label"] == 0)
        print(f"  {split:>12s}: {len(data):4d} exemples  (pos={n_pos}, neg={n_neg})")

    return subsets


class IMDbDataset(Dataset):
    """Dataset PyTorch pour IMDb avec tokenisation HuggingFace."""
    def __init__(self, examples, tokenizer, max_length=256):
        self.labels = [ex["label"] for ex in examples]
        texts = [ex["text"] for ex in examples]
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def get_dataloaders(subsets, tokenizer, batch_size=16, max_length=256):
    """
    Crée les DataLoaders pour chaque split.
    Returns:
        dict avec clés 'train', 'validation', 'test'
    """
    loaders = {}
    for split, examples in subsets.items():
        ds = IMDbDataset(examples, tokenizer, max_length)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
    return loaders


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    data = load_imdb_subset(50, 20, 30)
    loaders = get_dataloaders(data, tok, batch_size=8)
    batch = next(iter(loaders["train"]))
    print("input_ids:", batch["input_ids"].shape, "labels:", batch["labels"])
