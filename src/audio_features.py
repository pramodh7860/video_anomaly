"""
Audio branch (Section 4 of the spec).

    Raw audio -> Mel-spectrogram -> Audio CNN -> Audio feature vector

The Mel-spectrogram is only a representation of the audio (Section 16);
it does NOT classify anything. The small Audio CNN below learns the actual
discriminative patterns. No pretrained audio transformer is used, per spec.
"""
import numpy as np
import librosa
import torch
import torch.nn as nn


def extract_mel_spectrogram(
    audio_snippet: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """audio_snippet: 1D waveform for ONE temporal snippet.
    Returns a (n_mels, T_frames) log-mel-spectrogram."""
    if audio_snippet.size == 0:
        # silent/missing audio -> zero spectrogram of expected shape
        n_frames = 1 + sample_rate // hop_length
        return np.zeros((n_mels, n_frames), dtype=np.float32)

    mel = librosa.feature.melspectrogram(
        y=audio_snippet.astype(np.float32),
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


class AudioCNN(nn.Module):
    """A small CNN over the mel-spectrogram of one snippet.
    Learns patterns FROM the mel-spectrogram -- the spectrogram itself is
    just the input representation, per Section 16."""

    def __init__(self, n_mels: int = 64, feat_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, feat_dim)
        self.feat_dim = feat_dim

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, 1, n_mels, T_frames) -> (B, feat_dim)"""
        x = self.conv(mel)
        x = x.flatten(1)
        return self.fc(x)


def normalize_mel(mel: np.ndarray) -> np.ndarray:
    """Per-snippet standardization (zero mean, unit std) for stable training."""
    mean, std = mel.mean(), mel.std() + 1e-6
    return (mel - mean) / std
