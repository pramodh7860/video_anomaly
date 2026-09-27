"""
Precomputes, for every video, a temporally-aligned sequence of:
    - visual features (frozen MobileNetV3)   -> visual.npy  (T, Dv)
    - mel-spectrograms (fixed representation) -> mel.npy    (T, n_mels, frames)

Both arrays share the same T and the same snippet boundaries
[i*snippet_seconds, (i+1)*snippet_seconds), i = 0..T-1, so index i in one
array always corresponds to index i in the other (Section 5).

Usage:
    python scripts/extract_features.py --config configs/config.yaml \
        --video_dir data/raw/videos --out_dir data/processed
"""
import argparse
import os
import sys

import numpy as np
import librosa
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_config, get_device
from src.video_features import MobileNetV3FeatureExtractor, read_snippet_frames
from src.audio_features import extract_mel_spectrogram, normalize_mel


def get_video_duration(path: str) -> float:
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frame_count / fps if fps > 0 else 0.0


def extract_one_video(video_path: str, out_dir: str, extractor, cfg, device):
    snippet_s = cfg["data"]["snippet_seconds"]
    n_frames = cfg["data"]["frames_per_snippet"]
    sr = cfg["data"]["sample_rate"]
    n_mels = cfg["data"]["n_mels"]
    max_snippets = cfg["data"]["max_snippets"]

    duration = get_video_duration(video_path)
    n_snippets = min(max_snippets, max(1, int(duration // snippet_s)))

    # Load full audio track once (if present) and slice per snippet, so audio
    # and video snippets are guaranteed to come from the same time interval.
    audio_wave = None
    try:
        audio_wave, _ = librosa.load(video_path, sr=sr, mono=True)
    except Exception:
        audio_wave = None  # video has no usable audio track

    visual_feats, mel_specs = [], []
    for i in range(n_snippets):
        start_s, end_s = i * snippet_s, (i + 1) * snippet_s

        frames = read_snippet_frames(video_path, start_s, end_s, n_frames=n_frames).to(device)
        with torch.no_grad():
            v_feat = extractor.extract_snippet(frames).cpu().numpy()
        visual_feats.append(v_feat)

        if audio_wave is not None:
            s0, s1 = int(start_s * sr), int(end_s * sr)
            snippet_audio = audio_wave[s0:s1]
        else:
            snippet_audio = np.array([], dtype=np.float32)
        mel = extract_mel_spectrogram(snippet_audio, sample_rate=sr, n_mels=n_mels)
        mel = normalize_mel(mel)
        mel_specs.append(mel)

    # pad mel frames to a common width within this video (librosa output width
    # can vary by a frame at snippet boundaries)
    max_w = max(m.shape[1] for m in mel_specs)
    mel_specs = [
        np.pad(m, ((0, 0), (0, max_w - m.shape[1])), mode="constant") for m in mel_specs
    ]

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "visual.npy"), np.stack(visual_feats).astype(np.float32))
    np.save(os.path.join(out_dir, "mel.npy"), np.stack(mel_specs).astype(np.float32))
    return len(visual_feats), audio_wave is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--video_dir", required=True, help="Folder of raw .mp4 videos")
    parser.add_argument("--out_dir", required=True, help="Where per-video feature folders go")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["train"]["device"])

    extractor = MobileNetV3FeatureExtractor(
        variant=cfg["model"]["visual_backbone"], pretrained=True
    ).to(device).eval()

    videos = []
    for root, _, files in os.walk(args.video_dir):
        for fname in files:
            if fname.lower().endswith((".mp4", ".avi", ".mkv")):
                video_path = os.path.join(root, fname)
                video_id = os.path.splitext(os.path.relpath(video_path, args.video_dir))[0]
                videos.append((video_id, video_path))
    no_audio_count = 0
    for video_id, video_path in tqdm(videos, desc="Extracting features"):
        out_dir = os.path.join(args.out_dir, video_id)
        if os.path.exists(os.path.join(out_dir, "visual.npy")):
            continue  # already extracted
        n, has_audio = extract_one_video(
            video_path, out_dir, extractor, cfg, device
        )
        if not has_audio:
            no_audio_count += 1

    if no_audio_count:
        print(
            f"[WARNING] {no_audio_count}/{len(videos)} videos had no usable audio track. "
            f"Their mel-spectrograms are zero-filled -- see README Section 'Dataset audio "
            f"coverage' before running the multimodal experiment (Section 13 of the spec: "
            f"do not silently fabricate audio)."
        )


if __name__ == "__main__":
    main()
