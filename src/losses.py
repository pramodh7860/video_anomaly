"""
Weakly-supervised MIL loss (Section 9-10 of the spec).

Each video is a "bag" with only a video-level label (0=normal, 1=anomaly);
snippets are unlabeled "instances". We use the standard MIL ranking loss
(Sultani et al.-style) that only ever sees:
    - the top-k snippet scores of normal videos
    - the top-k snippet scores of anomalous videos
    - the video-level label
It never receives, and never needs, any frame-level label.
"""
import torch
import torch.nn.functional as F


def mil_ranking_loss(
    snippet_scores: torch.Tensor,   # (B, T) in [0,1]
    labels: torch.Tensor,           # (B,) 0=normal, 1=anomaly
    mask: torch.Tensor = None,      # (B, T) bool, True = valid snippet
    topk_ratio: float = 0.15,
    margin: float = 1.0,
    smoothness_lambda: float = 8e-5,
    sparsity_lambda: float = 8e-5,
):
    """
    For every (normal, anomaly) pair in the batch, requires:
        mean(top-k scores of anomaly video) - mean(top-k scores of normal video) >= margin
    via a hinge loss, plus:
      - temporal smoothness: adjacent snippet scores shouldn't jump wildly
        (mainly enforced on anomalous videos, since normal videos should be
         uniformly low)
      - sparsity: only a few snippets of an anomalous video should actually
        fire, matching the assumption that anomalies are localized in time.
    """
    device = snippet_scores.device
    B, T = snippet_scores.shape
    normal_idx = (labels == 0).nonzero(as_tuple=True)[0]
    anomaly_idx = (labels == 1).nonzero(as_tuple=True)[0]

    def topk_mean(row_scores, row_mask):
        valid = row_scores[row_mask] if row_mask is not None else row_scores
        if valid.numel() == 0:
            return torch.tensor(0.0, device=device)
        k = max(1, int(round(topk_ratio * valid.numel())))
        return torch.topk(valid, k).values.mean()

    # ---- ranking hinge loss over all normal/anomaly pairs ----
    ranking_terms = []
    for a in anomaly_idx:
        a_mask = mask[a] if mask is not None else None
        a_topk = topk_mean(snippet_scores[a], a_mask)
        for n in normal_idx:
            n_mask = mask[n] if mask is not None else None
            n_topk = topk_mean(snippet_scores[n], n_mask)
            ranking_terms.append(F.relu(margin - (a_topk - n_topk)))
    ranking_loss = (
        torch.stack(ranking_terms).mean()
        if ranking_terms
        else torch.tensor(0.0, device=device, requires_grad=True)
    )

    # ---- smoothness + sparsity regularizers (computed on anomaly videos) ----
    smooth_terms, sparse_terms = [], []
    for a in anomaly_idx:
        s = snippet_scores[a]
        if mask is not None:
            s = s[mask[a]]
        if s.numel() > 1:
            smooth_terms.append(((s[1:] - s[:-1]) ** 2).mean())
        if s.numel() > 0:
            sparse_terms.append(s.mean())

    smooth_loss = torch.stack(smooth_terms).mean() if smooth_terms else torch.tensor(0.0, device=device)
    sparse_loss = torch.stack(sparse_terms).mean() if sparse_terms else torch.tensor(0.0, device=device)

    total = ranking_loss + smoothness_lambda * smooth_loss + sparsity_lambda * sparse_loss
    return total, {
        "ranking_loss": ranking_loss.item(),
        "smooth_loss": smooth_loss.item() if torch.is_tensor(smooth_loss) else smooth_loss,
        "sparse_loss": sparse_loss.item() if torch.is_tensor(sparse_loss) else sparse_loss,
    }
