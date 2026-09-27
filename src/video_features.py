"""
Video branch (Section 3 of the spec).

    Video frames -> MobileNetV3 -> Visual feature vector

MobileNetV3 is used ONLY as a feature extractor. It never outputs a
normal/anomaly decision -- its classification head is removed and replaced
with global average pooling, giving one fixed-length embedding per snippet.
Feature extraction (this file) is kept fully separate from the trainable
anomaly-detection network (model.py).
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


class MobileNetV3FeatureExtractor(nn.Module):
    """Wraps a pretrained (ImageNet) MobileNetV3 and strips its classifier
    so it returns a pooled feature vector per frame."""

    def __init__(self, variant: str = "mobilenet_v3_small", pretrained: bool = True):
        super().__init__()
        if variant == "mobilenet_v3_small":
            backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.feat_dim = 576  # output channels of backbone.features on v3-small
        elif variant == "mobilenet_v3_large":
            backbone = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.feat_dim = 960
        else:
            raise ValueError(f"Unknown MobileNetV3 variant: {variant}")

        self.features = backbone.features          # conv trunk only, no classifier
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Feature extraction is frozen by default -- it is not part of the
        # anomaly-detection network being trained (Section 3/16).
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) -> (B, feat_dim)"""
        feats = self.features(x)
        feats = self.pool(feats).flatten(1)
        return feats

    @torch.no_grad()
    def extract_snippet(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (T, 3, H, W) sampled from one snippet -> (feat_dim,)
        via mean pooling over the T frames' embeddings."""
        self.eval()
        feats = self.forward(frames)        # (T, feat_dim)
        return feats.mean(dim=0)            # (feat_dim,)


IMAGENET_TRANSFORM = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def read_snippet_frames(video_path: str, start_sec: float, end_sec: float,
                         n_frames: int = 16) -> torch.Tensor:
    """Uniformly samples n_frames frames from [start_sec, end_sec) of a video
    and returns them as a preprocessed tensor (n_frames, 3, 224, 224)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(start_sec * fps)
    end_frame = max(start_frame + 1, int(end_sec * fps))
    indices = np.linspace(start_frame, end_frame - 1, n_frames).astype(int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            # pad with the last valid frame (or black frame if none yet)
            frame = frames[-1].numpy().transpose(1, 2, 0) if frames else np.zeros((224, 224, 3), np.uint8)
            frame = (frame * 255).astype(np.uint8) if frames else frame
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(IMAGENET_TRANSFORM(frame))
    cap.release()
    return torch.stack(frames, dim=0)  # (n_frames, 3, 224, 224)
