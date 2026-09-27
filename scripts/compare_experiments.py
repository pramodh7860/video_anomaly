"""
Runs evaluate() for experiment1/2/3 checkpoints (if present) and prints a
single comparison table, matching Section 14's requirement to compare:
    1. Visual-only baseline
    2. Visual + attention
    3. Visual + audio + attention

Usage:
    python scripts/compare_experiments.py --config configs/config.yaml
"""
import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_config, get_device
from src.dataset import SnippetFeatureDataset, collate_batch
from src.train import FullModel
from src.evaluate import evaluate

NAMES = {
    1: "Exp1: Visual only",
    2: "Exp2: Visual + Attention",
    3: "Exp3: Visual + Audio + Attention (final)",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["train"]["device"])
    ckpt_dir = args.checkpoint_dir or cfg["train"]["checkpoint_dir"]

    test_ds = SnippetFeatureDataset(
        cfg["data"]["annotation_test"], cfg["data"]["root"], cfg["data"]["max_snippets"]
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_batch)

    rows = []
    for exp_id, name in NAMES.items():
        ckpt_path = os.path.join(ckpt_dir, f"experiment{exp_id}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[skip] {ckpt_path} not found -- run: python -m src.train --experiment {exp_id}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        model = FullModel(cfg, **ckpt["flags"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        metrics = evaluate(model, test_loader, device, cfg["online_inference"]["score_threshold"])
        metrics["Experiment"] = name
        rows.append(metrics)

    if not rows:
        print("No checkpoints found yet -- train experiments 1, 2, and 3 first.")
        return

    df = pd.DataFrame(rows).set_index("Experiment")
    df = df[["AUC", "Average_Precision", "Precision", "Recall", "F1",
             "avg_latency_ms_per_snippet", "throughput_snippets_per_sec"]]
    print(df.round(4).to_string())
    out_path = "results/experiment_comparison.csv"
    os.makedirs("results", exist_ok=True)
    df.to_csv(out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
