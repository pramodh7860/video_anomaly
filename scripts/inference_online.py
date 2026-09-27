"""
Online / near-real-time inference (Section 11).

Processes a video as a sequential stream of snippets rather than requiring
the whole video up front. A sliding window of the most recent
`window_snippets` snippets is kept in memory; each time a new snippet
arrives, the window (visual+audio -> fusion -> GRU -> attention -> score) is
re-scored and the score for the newest snippet is emitted immediately. This
keeps every forward pass causal (no access to future snippets) while still
letting attention compare the newest snippet against its recent context, as
required by Section 8.

This script measures and reports actual wall-clock latency/FPS per snippet
(Section 14) rather than assuming real-time performance (Section 11).

Usage:
    python scripts/inference_online.py --config configs/config.yaml \
        --checkpoint checkpoints/experiment3_best.pt --video path/to/video.mp4
"""
import argparse
import os
import sys
import time
from collections import deque

import librosa
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_config, get_device, moving_average
from src.video_features import MobileNetV3FeatureExtractor, read_snippet_frames
from src.audio_features import extract_mel_spectrogram, normalize_mel
from src.train import FullModel


class OnlineAnomalyDetector:
    def __init__(self, cfg, checkpoint_path, device):
        self.cfg = cfg
        self.device = device
        self.window = cfg["data"]["max_snippets"]  # cap window length for memory/latency
        self.window = min(self.window, 32)          # practical online window (Section 11)
        self.snippet_s = cfg["data"]["snippet_seconds"]
        self.threshold = cfg["online_inference"]["score_threshold"]
        self.smoothing = cfg["online_inference"]["smoothing_window"]

        ckpt = torch.load(checkpoint_path, map_location=device)
        self.flags = ckpt["flags"]
        self.model = FullModel(cfg, **self.flags).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.visual_extractor = MobileNetV3FeatureExtractor(
            variant=cfg["model"]["visual_backbone"], pretrained=True
        ).to(device).eval()

        self.visual_buf = deque(maxlen=self.window)
        self.mel_buf = deque(maxlen=self.window)
        self.score_history = []
        self.latencies = []

    @torch.no_grad()
    def process_snippet(self, frames: torch.Tensor, audio_snippet: np.ndarray) -> float:
        """frames: (n_frames, 3, 224, 224) for this snippet.
        audio_snippet: 1D waveform for this snippet.
        Returns the (smoothed) anomaly score for the newest snippet."""
        t0 = time.perf_counter()

        v_feat = self.visual_extractor.extract_snippet(frames.to(self.device)).cpu().numpy()
        mel = extract_mel_spectrogram(
            audio_snippet, sample_rate=self.cfg["data"]["sample_rate"],
            n_mels=self.cfg["data"]["n_mels"],
        )
        mel = normalize_mel(mel)

        self.visual_buf.append(v_feat)
        self.mel_buf.append(mel)

        # pad mel widths within current window to a common size
        max_w = max(m.shape[1] for m in self.mel_buf)
        mels = [np.pad(m, ((0, 0), (0, max_w - m.shape[1]))) for m in self.mel_buf]

        visual = torch.from_numpy(np.stack(self.visual_buf)).float().unsqueeze(0).to(self.device)
        mel_t = torch.from_numpy(np.stack(mels)).float().unsqueeze(0).to(self.device)
        mask = torch.ones(1, visual.shape[1], dtype=torch.bool, device=self.device)

        scores, _ = self.model(visual, mel_t, mask)
        newest_score = scores[0, -1].item()

        elapsed = time.perf_counter() - t0
        self.latencies.append(elapsed)
        self.score_history.append(newest_score)

        smoothed = moving_average(np.array(self.score_history), self.smoothing)[-1]
        return float(smoothed)

    def report_speed(self):
        arr = np.array(self.latencies)
        return {
            "avg_latency_ms": arr.mean() * 1000,
            "p95_latency_ms": np.percentile(arr, 95) * 1000,
            "fps": 1.0 / arr.mean(),
        }


def stream_video(video_path: str, cfg, detector: OnlineAnomalyDetector):
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()

    snippet_s = cfg["data"]["snippet_seconds"]
    n_frames = cfg["data"]["frames_per_snippet"]
    sr = cfg["data"]["sample_rate"]
    n_snippets = max(1, int(duration // snippet_s))

    audio_wave = None
    try:
        audio_wave, _ = librosa.load(video_path, sr=sr, mono=True)
    except Exception:
        pass

    anomaly_intervals = []
    in_anomaly = False
    interval_start = None

    print(f"Streaming {n_snippets} snippets ({snippet_s}s each)...")
    for i in range(n_snippets):
        start_s, end_s = i * snippet_s, (i + 1) * snippet_s
        frames = read_snippet_frames(video_path, start_s, end_s, n_frames=n_frames)
        if audio_wave is not None:
            s0, s1 = int(start_s * sr), int(end_s * sr)
            audio_snippet = audio_wave[s0:s1]
        else:
            audio_snippet = np.array([], dtype=np.float32)

        score = detector.process_snippet(frames, audio_snippet)
        label = "ANOMALY" if score >= detector.threshold else "normal"
        print(f"  [{start_s:6.1f}s - {end_s:6.1f}s] score={score:.3f}  {label}")

        if score >= detector.threshold and not in_anomaly:
            in_anomaly, interval_start = True, start_s
        elif score < detector.threshold and in_anomaly:
            in_anomaly = False
            anomaly_intervals.append((interval_start, end_s))
    if in_anomaly:
        anomaly_intervals.append((interval_start, n_snippets * snippet_s))

    speed = detector.report_speed()
    print("\n--- Summary ---")
    print(f"Prediction: {'ANOMALY' if anomaly_intervals else 'NORMAL'}")
    for s, e in anomaly_intervals:
        print(f"  Approx. anomalous interval: {s:.1f}s - {e:.1f}s")
    print(f"Avg latency/snippet: {speed['avg_latency_ms']:.1f} ms "
          f"(p95: {speed['p95_latency_ms']:.1f} ms), ~{speed['fps']:.2f} snippets/sec processed")
    print("Note: this reports MEASURED speed only; whether it qualifies as "
          "'real-time' depends on your snippet length vs. this latency (Section 11).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["train"]["device"])
    detector = OnlineAnomalyDetector(cfg, args.checkpoint, device)
    stream_video(args.video, cfg, detector)


if __name__ == "__main__":
    main()
