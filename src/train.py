"""
Training loop for the MIL weakly-supervised model (Section 9, 12).

Run one of the three experiments via --experiment {1,2,3}:
    1 -> visual only, no attention
    2 -> visual only, + attention
    3 -> visual + audio + attention   (final model)

Usage:
    python -m src.train --config configs/config.yaml --experiment 3
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_config, set_seed, get_device
from src.dataset import SnippetFeatureDataset, collate_batch
from src.model import MultimodalAnomalyNet
from src.audio_features import AudioCNN
from src.losses import mil_ranking_loss


EXPERIMENT_FLAGS = {
    1: dict(use_audio=False, use_attention=False),
    2: dict(use_audio=False, use_attention=True),
    3: dict(use_audio=True, use_attention=True),
}


class FullModel(torch.nn.Module):
    """Wraps AudioCNN + MultimodalAnomalyNet so audio features are computed
    from cached mel-spectrograms on the fly (AudioCNN is trainable, unlike
    the frozen MobileNetV3 features which are already precomputed)."""

    def __init__(self, cfg, use_audio: bool, use_attention: bool):
        super().__init__()
        self.use_audio = use_audio
        if use_audio:
            self.audio_cnn = AudioCNN(
                n_mels=cfg["data"]["n_mels"], feat_dim=cfg["model"]["audio_feat_dim"]
            )
        self.core = MultimodalAnomalyNet(
            visual_feat_dim=cfg["model"]["visual_feat_dim"],
            audio_feat_dim=cfg["model"]["audio_feat_dim"],
            gru_hidden_dim=cfg["model"]["gru_hidden_dim"],
            gru_layers=cfg["model"]["gru_layers"],
            attention_dim=cfg["model"]["attention_dim"],
            use_audio=use_audio,
            use_attention=use_attention,
            dropout=cfg["model"]["dropout"],
        )

    def forward(self, visual, mel=None, mask=None):
        audio_feats = None
        if self.use_audio:
            B, T, n_mels, frames = mel.shape
            mel_flat = mel.view(B * T, 1, n_mels, frames)
            audio_feats = self.audio_cnn(mel_flat).view(B, T, -1)
        scores, attn, _ = self.core(visual, audio_feats, mask)
        return scores, attn


def run_epoch(model, loader, optimizer, cfg, device, train: bool = True):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0

    for batch in tqdm(loader, desc="train" if train else "eval", leave=False):
        visual = batch["visual"].to(device)
        mel = batch["mel"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(train):
            scores, _ = model(visual, mel, mask)
            loss, parts = mil_ranking_loss(
                scores, labels, mask,
                topk_ratio=cfg["train"]["topk_ratio"],
                margin=cfg["train"]["margin"],
                smoothness_lambda=cfg["train"]["smoothness_lambda"],
                sparsity_lambda=cfg["train"]["sparsity_lambda"],
            )
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = get_device(cfg["train"]["device"])
    flags = EXPERIMENT_FLAGS[args.experiment]

    train_ds = SnippetFeatureDataset(
        cfg["data"]["annotation_train"], cfg["data"]["root"], cfg["data"]["max_snippets"]
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_batch
    )

    model = FullModel(cfg, **flags).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
    )

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, optimizer, cfg, device, train=True)
        print(f"[Exp {args.experiment}] Epoch {epoch}/{cfg['train']['epochs']} "
              f"train_loss={train_loss:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            ckpt_path = os.path.join(
                cfg["train"]["checkpoint_dir"], f"experiment{args.experiment}_best.pt"
            )
            torch.save({"model_state": model.state_dict(), "flags": flags, "epoch": epoch}, ckpt_path)

    print(f"Training done. Best checkpoint: experiment{args.experiment}_best.pt")


if __name__ == "__main__":
    main()
