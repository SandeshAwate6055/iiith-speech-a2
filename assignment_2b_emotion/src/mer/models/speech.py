"""Speech-only Emotion Recognition model (S*): SSL layer sum -> projection -> temporal -> pooling -> MLP."""
import torch
import torch.nn as nn

from mer.models.blocks import AttentiveStatsPool, LayerWeightedSum, MeanPool, MLPHead, TemporalEncoder


class SpeechBranch(nn.Module):
    """Encodes cached SSL hidden states [B, L, T, D] into frame-level and utterance-level embeddings."""

    def __init__(
        self,
        num_layers: int = 13,
        in_dim: int = 768,
        proj_dim: int = 256,
        temporal_arch: str = "gru",
        temporal_hidden: int = 128,
        temporal_layers: int = 1,
        pooling: str = "asp",
        dropout: float = 0.3,
        spec_mask_prob: float = 0.0,
    ):
        super().__init__()
        self.layer_sum = LayerWeightedSum(num_layers)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )
        self.temporal = TemporalEncoder(proj_dim, temporal_hidden, arch=temporal_arch,
                                        num_layers=temporal_layers, dropout=dropout)
        self.frame_dim = self.temporal.out_dim
        if pooling == "asp":
            self.pool = AttentiveStatsPool(self.frame_dim)
        elif pooling == "mean":
            self.pool = MeanPool(self.frame_dim)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        self.out_dim = self.pool.out_dim
        self.spec_mask_prob = spec_mask_prob

    def _spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.spec_mask_prob <= 0:
            return x
        B, T, D = x.shape
        x = x.clone()
        for b in range(B):
            if torch.rand(1).item() < self.spec_mask_prob and T > 10:
                t_len = int(torch.randint(2, max(3, T // 6), (1,)).item())
                t0 = int(torch.randint(0, T - t_len, (1,)).item())
                x[b, t0 : t0 + t_len, :] = 0.0
            if torch.rand(1).item() < self.spec_mask_prob:
                f_len = int(torch.randint(4, max(5, D // 8), (1,)).item())
                f0 = int(torch.randint(0, D - f_len, (1,)).item())
                x[b, :, f0 : f0 + f_len] = 0.0
        return x

    def forward(self, ssl_feats: torch.Tensor, mask: torch.Tensor):
        # ssl_feats: [B, L, T, D], mask: [B, T] (bool)
        x, lw = self.layer_sum(ssl_feats)
        x = self._spec_augment(x)
        x = self.proj(x)
        frames = self.temporal(x, mask)
        emb, attn_w = self.pool(frames, mask)
        return {"frames": frames, "emb": emb, "attn_weights": attn_w, "layer_weights": lw}


class SpeechEmotionModel(nn.Module):
    def __init__(self, num_classes: int = 7, head_hidden: int = 256, dropout: float = 0.3, **branch_kwargs):
        super().__init__()
        self.branch = SpeechBranch(dropout=dropout, **branch_kwargs)
        self.head = MLPHead(self.branch.out_dim, head_hidden, num_classes, dropout)

    def forward(self, ssl_feats: torch.Tensor, mask: torch.Tensor, **_):
        out = self.branch(ssl_feats, mask)
        logits = self.head(out["emb"])
        return {
            "logits": logits,
            "temporal_emb": out["emb"],
            "attn_weights": out["attn_weights"],
            "layer_weights": out["layer_weights"],
        }
