"""
Evaluation (Section 14).

Computes, on the held-out test set:
    - AUC, Average Precision (video-level anomaly detection quality)
    - Precision, Recall, F1 at a fixed decision threshold
    - Inference latency (ms/snippet) and processing FPS, for the online
      objective

Video-level score = model.video_level_score() (top-k mean of snippet
scores), matching how the model is trained and how online inference
aggregates results.

Usage:
    python -m src.evaluate --config configs/config.yaml \
        --checkpoint checkpoints/experiment3_best.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_config, get_device
from src.dataset import SnippetFeatureDataset, collate_batch
from src.train import FullModel


def evaluate(model, loader, device, threshold: float = 0.5):
    model.eval()
    y_true, y_score = [], []
    latencies = []

    with torch.no_grad():
        for batch in loader:
            visual = batch["visual"].to(device)
            mel = batch["mel"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].numpy()

            t0 = time.perf_counter()
            scores, _ = model(visual, mel, mask)
            video_scores = model.core.video_level_score(scores, mask=mask).cpu().numpy()
            elapsed = time.perf_counter() - t0

            n_snippets_total = mask.sum().item()
            latencies.append(elapsed / max(1, n_snippets_total))

            y_true.extend(labels.tolist())
            y_score.extend(video_scores.tolist())

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "AUC": roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan"),
        "Average_Precision": average_precision_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "avg_latency_ms_per_snippet": float(np.mean(latencies)) * 1000,
        "throughput_snippets_per_sec": 1.0 / (float(np.mean(latencies)) + 1e-9),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["train"]["device"])

    ckpt = torch.load(args.checkpoint, map_location=device)
    flags = ckpt["flags"]

    test_ds = SnippetFeatureDataset(
        cfg["data"]["annotation_test"], cfg["data"]["root"], cfg["data"]["max_snippets"]
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_batch)

    model = FullModel(cfg, **flags).to(device)
    model.load_state_dict(ckpt["model_state"])

    metrics = evaluate(model, test_loader, device, threshold=cfg["online_inference"]["score_threshold"])
    print(f"Results for {args.checkpoint} (use_audio={flags['use_audio']}, "
          f"use_attention={flags['use_attention']}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
