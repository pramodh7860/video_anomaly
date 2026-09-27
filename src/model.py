"""
Core trainable network (Sections 6-10 of the spec):

    Visual feature [+ Audio feature] -> Concatenation -> Fused Feature
    Fused Feature sequence -> GRU -> Attention -> Snippet Anomaly Scores

This single class implements all three experiments from Section 12 via two
flags (use_audio, use_attention), so the contribution of each component can
be measured directly instead of duplicating code per experiment:

    Experiment 1 (visual baseline):      use_audio=False, use_attention=False
    Experiment 2 (+ attention):          use_audio=False, use_attention=True
    Experiment 3 (final, + audio):       use_audio=True,  use_attention=True

The GRU and attention module never see the video-level label directly --
they only ever receive feature sequences. Supervision enters solely through
the MIL loss computed on the output anomaly scores (see losses.py).
"""
import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Learns a scalar importance weight per snippet from the GRU hidden
    states. Weights are produced purely by backpropagation -- nothing here
    is hand-specified (Section 8)."""

    def __init__(self, hidden_dim: int, attn_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, gru_out: torch.Tensor, mask: torch.Tensor = None):
        """gru_out: (B, T, H) -> weighted (B, T, H), weights (B, T)"""
        scores = self.proj(gru_out).squeeze(-1)          # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)            # (B, T)
        weighted = gru_out * weights.unsqueeze(-1)         # (B, T, H)
        return weighted, weights


class MultimodalAnomalyNet(nn.Module):
    def __init__(
        self,
        visual_feat_dim: int = 576,
        audio_feat_dim: int = 128,
        gru_hidden_dim: int = 256,
        gru_layers: int = 1,
        attention_dim: int = 128,
        use_audio: bool = True,
        use_attention: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.use_audio = use_audio
        self.use_attention = use_attention

        fused_dim = visual_feat_dim + (audio_feat_dim if use_audio else 0)

        self.input_proj = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=fused_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )

        if use_attention:
            self.attention = TemporalAttention(gru_hidden_dim, attention_dim)

        self.score_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, gru_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, visual_feats: torch.Tensor, audio_feats: torch.Tensor = None,
                mask: torch.Tensor = None, hidden_state: torch.Tensor = None):
        """
        visual_feats: (B, T, visual_feat_dim)
        audio_feats:  (B, T, audio_feat_dim)      -- required if use_audio
        mask:         (B, T) bool, True = valid snippet (for padded videos)
        hidden_state: optional GRU hidden state for sequential/online use

        Returns:
            scores: (B, T) snippet-level anomaly scores in [0, 1]
            attn_weights: (B, T) or None
            new_hidden: final GRU hidden state (for streaming continuation)
        """
        if self.use_audio:
            assert audio_feats is not None, "use_audio=True but no audio features were given"
            fused = torch.cat([visual_feats, audio_feats], dim=-1)   # (B, T, fused_dim)
        else:
            fused = visual_feats

        fused = self.input_proj(fused)
        gru_out, new_hidden = self.gru(fused, hidden_state)          # (B, T, H)

        attn_weights = None
        if self.use_attention:
            gru_out, attn_weights = self.attention(gru_out, mask)

        scores = self.score_head(gru_out).squeeze(-1)                 # (B, T)
        if mask is not None:
            scores = scores * mask.float()
        return scores, attn_weights, new_hidden

    def video_level_score(self, snippet_scores: torch.Tensor, topk_ratio: float = 0.15,
                           mask: torch.Tensor = None) -> torch.Tensor:
        """Aggregates snippet scores into one video-level anomaly score via
        mean of the top-k most anomalous snippets (standard MIL aggregation --
        Section 10)."""
        B, T = snippet_scores.shape
        video_scores = []
        for b in range(B):
            s = snippet_scores[b]
            if mask is not None:
                s = s[mask[b]]
            if s.numel() == 0:
                video_scores.append(torch.tensor(0.0, device=snippet_scores.device))
                continue
            k = max(1, int(round(topk_ratio * s.numel())))
            topk = torch.topk(s, k).values
            video_scores.append(topk.mean())
        return torch.stack(video_scores)
