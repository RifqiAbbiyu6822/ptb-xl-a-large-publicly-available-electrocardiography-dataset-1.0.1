"""
engine.py
Loop training/evaluasi per epoch. Dipisah dari train.py supaya train.py
fokus ke orkestrasi (argparse, checkpoint, early stopping).
"""

import torch
import numpy as np
from tqdm import tqdm

from metrics import compute_multilabel_metrics


def train_one_epoch(model, loader, optimizer, criterion, device,
                     scaler=None, grad_clip: float = 1.0) -> float:
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="train", leave=False)
    for signals, labels in pbar:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(signals)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * signals.size(0)
        pbar.set_postfix(loss=f"{batch_loss:.4f}")

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, target_classes=None,
             threshold: float = 0.5) -> dict:
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for signals, labels in tqdm(loader, desc="eval", leave=False):
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(signals)
        loss = criterion(logits, labels)
        total_loss += loss.item() * signals.size(0)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_multilabel_metrics(all_labels, all_probs, threshold, target_classes)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics
