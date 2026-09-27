"""
Dataset loader.

Design note on what is precomputed vs. trained online:
  - MobileNetV3 is FROZEN (Section 3), so its per-snippet visual features are
    precomputed once by scripts/extract_features.py and cached as .npy. This
    just saves redundant compute -- MobileNetV3 itself never changes.
  - The mel-spectrogram is a fixed signal representation (Section 16, not
    trainable), so it is also precomputed and cached.
  - The Audio CNN, GRU, attention, and scoring head ARE trainable, so they
    always run live on top of these cached inputs during training/inference.

Each dataset item = one video (one MIL "bag"):
    visual_feats: (T, visual_feat_dim)
    mel_specs:    (T, n_mels, frames)   -- per-snippet mel-spectrogram
    label:        0 (normal) or 1 (anomaly)
Snippet index i in visual_feats and mel_specs both correspond to the SAME
[i*snippet_seconds, (i+1)*snippet_seconds) interval (Section 5: temporal
alignment is enforced at feature-extraction time, see extract_features.py).
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SnippetFeatureDataset(Dataset):
    def __init__(self, annotation_csv: str, features_root: str, max_snippets: int = 200):
        """
        annotation_csv columns: video_id,label   (label in {0,1})
        features_root/<video_id>/visual.npy  -> (T, visual_feat_dim)
        features_root/<video_id>/mel.npy     -> (T, n_mels, frames)
        """
        self.df = pd.read_csv(annotation_csv)
        self.features_root = features_root
        self.max_snippets = max_snippets

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id, label = row["video_id"], int(row["label"])
        vdir = os.path.join(self.features_root, str(video_id))

        visual = np.load(os.path.join(vdir, "visual.npy"))   # (T, Dv)
        mel = np.load(os.path.join(vdir, "mel.npy"))         # (T, n_mels, frames)

        T = visual.shape[0]
        T_use = min(T, self.max_snippets)
        visual = visual[:T_use]
        mel = mel[:T_use]

        pad = self.max_snippets - T_use
        mask = np.ones(self.max_snippets, dtype=bool)
        if pad > 0:
            visual = np.concatenate([visual, np.zeros((pad, visual.shape[1]), dtype=np.float32)], axis=0)
            mel = np.concatenate([mel, np.zeros((pad,) + mel.shape[1:], dtype=np.float32)], axis=0)
            mask[T_use:] = False

        return {
            "video_id": video_id,
            "visual": torch.from_numpy(visual).float(),
            "mel": torch.from_numpy(mel).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "mask": torch.from_numpy(mask),
            "n_valid": T_use,
        }


def collate_batch(batch):
    return {
        "video_id": [b["video_id"] for b in batch],
        "visual": torch.stack([b["visual"] for b in batch]),
        "mel": torch.stack([b["mel"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "n_valid": [b["n_valid"] for b in batch],
    }
