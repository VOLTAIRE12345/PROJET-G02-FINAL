"""
model_setup.py — Groupe G02
Chargement de BERT-base-uncased avec adaptation CPU automatique.
Weight decay découplé : exclu sur bias et LayerNorm (pratique HuggingFace recommandée).
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "bert-base-uncased"
NUM_LABELS  = 2


def get_model_and_tokenizer(model_name=MODEL_NAME, num_labels=NUM_LABELS,
                             dropout_prob=0.1, device=None):
    """
    Charge BERT-base et son tokenizer.
    
    Args:
        model_name:   identifiant HuggingFace du modèle
        num_labels:   nombre de classes de sortie
        dropout_prob: probabilité de dropout appliquée à la tête de classification
                      (hidden_dropout_prob ET attention_probs_dropout_prob)
        device:       'cuda' ou 'cpu' (auto-détecté si None)
    Returns:
        (model, tokenizer)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        hidden_dropout_prob=dropout_prob,
        attention_probs_dropout_prob=dropout_prob,
        torch_dtype=torch.float32,   # float32 obligatoire pour CPU
    )
    model = model.to(device)

    # Optimisations threading pour CPU
    if device.type == "cpu":
        torch.set_num_threads(4)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres BERT : {n_params/1e6:.1f} M")
    return model, tokenizer


def build_optimizer(model, lr=2e-5, weight_decay=1e-4):
    """
    Construit AdamW avec weight decay découplé (pratique recommandée pour BERT).
    Le weight decay N'EST PAS appliqué aux biais et aux poids LayerNorm,
    conformément à la pratique standard HuggingFace (Devlin et al., 2019).
    
    Args:
        model:        modèle PyTorch
        lr:           learning rate
        weight_decay: pénalité L2 (appliquée uniquement aux poids hors biais/LN)
    Returns:
        optimizer AdamW configuré
    """
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        # Groupe 1 : poids régularisés
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        # Groupe 2 : biais + LayerNorm — PAS de weight decay
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    from torch.optim import AdamW
    return AdamW(grouped_params, lr=lr)


if __name__ == "__main__":
    model, tok = get_model_and_tokenizer(dropout_prob=0.1)
    opt = build_optimizer(model, lr=2e-5, weight_decay=1e-4)
    print("Modèle et optimiseur prêts.")
