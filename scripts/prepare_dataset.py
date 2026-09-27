"""
Builds video-level annotation CSVs (video_id,label) from a folder of raw
videos, plus a train/test split. No frame-level labels are ever produced or
used (Section 2 / 16).

Default naming rule matches XD-Violence's official convention, where each
filename encodes its label, e.g.:
    "A.Beyond.Skyline.2017__#00-01-05_00-01-12_label_A.mp4"   -> normal
    "A.Beyond.Skyline.2017__#00-08-17_00-08-30_label_B2-0-0.mp4" -> anomaly
i.e. filenames containing "_label_A" (with nothing else appended) are Normal;
every other "_label_*" is an anomaly category (Fighting/Riot/Abuse/Explosion/
Shooting/CarAccident) and is treated as Anomaly=1, since this project only
needs the binary video-level label (Section 1).

For a different dataset, replace `infer_label()` with the appropriate rule
and keep everything else the same.

Usage:
    python scripts/prepare_dataset.py --video_dir data/raw/videos \
        --out_dir data/annotations --test_ratio 0.2
"""
import argparse
import os
import random

import pandas as pd


def infer_label(filename: str, parent_dir: str = "") -> int:
    name = filename.lower()
    parent = parent_dir.lower()
    if "_label_a" in name or "__label_a" in name:
        return 0  # normal
    if "_label_" in name:
        return 1  # any other label_* code = an anomaly category
    # fallback heuristic for datasets that use class folders
    if "nonviolence" in parent or "non-violence" in parent or "normal" in parent:
        return 0
    if "violence" in parent or "anomaly" in parent:
        return 1
    if "normal" in name or "nonviolence" in name or "non-violence" in name:
        return 0
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    videos = []
    for root, _, files in os.walk(args.video_dir):
        for filename in files:
            if filename.lower().endswith((".mp4", ".avi", ".mkv")):
                relative_path = os.path.relpath(os.path.join(root, filename), args.video_dir)
                video_id = os.path.splitext(relative_path)[0]
                videos.append((video_id, filename, os.path.basename(root)))
    if not videos:
        raise SystemExit(f"No video files found in {args.video_dir}")

    rows = [
        {"video_id": video_id, "label": infer_label(filename, parent_dir)}
        for video_id, filename, parent_dir in videos
    ]
    df = pd.DataFrame(rows)

    n_normal, n_anom = (df.label == 0).sum(), (df.label == 1).sum()
    print(f"Found {len(df)} videos -> {n_normal} normal / {n_anom} anomaly")
    if n_normal == 0 or n_anom == 0:
        print("[WARNING] One class has zero videos -- check infer_label() for this dataset.")

    # simple stratified split
    df_shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test_frames, train_frames = [], []
    for label in (0, 1):
        subset = df_shuffled[df_shuffled.label == label]
        n_test = max(1, int(len(subset) * args.test_ratio)) if len(subset) > 0 else 0
        test_frames.append(subset.iloc[:n_test])
        train_frames.append(subset.iloc[n_test:])

    train_df = pd.concat(train_frames).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test_df = pd.concat(test_frames).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    train_df.to_csv(os.path.join(args.out_dir, "train_list.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test_list.csv"), index=False)
    print(f"Wrote {len(train_df)} train / {len(test_df)} test rows to {args.out_dir}")


if __name__ == "__main__":
    main()
