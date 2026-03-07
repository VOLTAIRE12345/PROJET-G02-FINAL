"""
train_eval.py — Groupe G02
Fonctions d'entraînement et d'évaluation pour BERT fine-tuning.
Inclut : early stopping, gradient clipping, scheduler linéaire avec warmup.
"""
import time
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ─── Scheduler linéaire avec warmup ──────────────────────────────────────────
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Scheduler de learning rate : montée linéaire sur warmup_steps,
    puis descente linéaire jusqu'à 0.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        return max(0.0, float(num_training_steps - current_step) /
                   max(1, num_training_steps - num_warmup_steps))
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


# ─── Entraînement une époque ──────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device, max_grad_norm=1.0):
    """
    Entraîne le modèle sur un epoch complet.
    Applique gradient clipping pour stabiliser l'entraînement.
    
    Returns:
        (avg_loss, accuracy) sur l'epoch
    """
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * batch["labels"].size(0)
        preds = outputs.logits.argmax(dim=-1)
        total_correct += (preds == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)

    return total_loss / total_samples, total_correct / total_samples


# ─── Évaluation ───────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    """
    Évalue le modèle sur un DataLoader.
    
    Returns:
        dict avec 'loss', 'accuracy', 'f1'
    """
    model.eval()
    all_preds, all_labels, total_loss, total_samples = [], [], 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item() * batch["labels"].size(0)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            total_samples += batch["labels"].size(0)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="binary")
    return {"loss": total_loss / total_samples, "accuracy": acc, "f1": f1}


# ─── Boucle d'entraînement complète avec early stopping ──────────────────────
def train_model(model, loaders, optimizer, num_epochs, warmup_ratio,
                device, max_grad_norm=1.0, patience=2, verbose=True):
    """
    Entraîne le modèle avec early stopping sur la val loss.
    Restaure automatiquement le meilleur état trouvé.
    
    Args:
        model:         modèle PyTorch
        loaders:       dict {'train', 'validation'} de DataLoaders
        optimizer:     optimiseur AdamW
        num_epochs:    nombre maximum d'époques
        warmup_ratio:  fraction des steps utilisés pour le warmup
        device:        device PyTorch
        max_grad_norm: norme maximale pour gradient clipping
        patience:      patience early stopping (en époques sans amélioration)
        verbose:       affiche les métriques à chaque époque
    Returns:
        history: dict avec listes 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'val_f1'
    """
    total_steps   = num_epochs * len(loaders["train"])
    warmup_steps  = int(warmup_ratio * total_steps)
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [], "val_f1": []
    }
    best_val_loss    = float("inf")
    epochs_no_improve = 0
    best_state       = None
    t0               = time.time()

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, loaders["train"], optimizer, scheduler, device, max_grad_norm
        )
        val_metrics = evaluate(model, loaders["validation"], device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        if verbose:
            print(f"  Epoch {epoch}/{num_epochs} | "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f}  "
                  f"val_acc={val_metrics['accuracy']:.4f}  "
                  f"val_f1={val_metrics['f1']:.4f}")

        # Early stopping : surveille la val_loss
        if val_metrics["loss"] < best_val_loss:
            best_val_loss    = val_metrics["loss"]
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  [Early stopping] époque {epoch} — patience={patience} atteinte.")
                break

    elapsed = time.time() - t0
    if verbose:
        print(f"  Durée entraînement : {elapsed:.1f}s")

    # Restaurer le meilleur checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    history["train_time"] = elapsed
    return history


if __name__ == "__main__":
    print("Module train_eval chargé avec succès.")
