"""Reusable neural building blocks for Speech, Text, and Multimodal Emotion Recognition."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerWeightedSum(nn.Module):
    """SUPERB-style learnable softmax-weighted sum over SSL hidden layers [B, L, T, D] -> [B, T, D]."""

    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, x: torch.Tensor):
        w = F.softmax(self.weights, dim=0)  # [L]
        out = torch.einsum("l,bltd->btd", w, x)
        return out, w


class AttentiveStatsPool(nn.Module):
    """Attention-weighted mean and standard deviation pooling (Okabe et al., Interspeech 2018)."""

    def __init__(self, in_dim: int, bottleneck: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, bottleneck),
            nn.Tanh(),
            nn.Linear(bottleneck, 1),
        )
        self.out_dim = in_dim * 2

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: [B, T, D], mask: [B, T] (True = valid)
        scores = self.attn(x).squeeze(-1)  # [B, T]
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)
        w = F.softmax(scores, dim=1)  # [B, T]
        mean = torch.sum(x * w.unsqueeze(-1), dim=1)  # [B, D]
        var = torch.sum(((x - mean.unsqueeze(1)) ** 2) * w.unsqueeze(-1), dim=1)
        std = torch.sqrt(var.clamp(min=1e-6))
        return torch.cat([mean, std], dim=-1), w


class MeanPool(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.out_dim = in_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is None:
            w = torch.full((x.size(0), x.size(1)), 1.0 / x.size(1), device=x.device)
            return x.mean(dim=1), w
        m = mask.float()
        denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        w = m / denom
        return torch.sum(x * w.unsqueeze(-1), dim=1), w


class AttnPool(nn.Module):
    """Single-head attention pooling over token sequences [B, N, D] -> [B, D]."""

    def __init__(self, in_dim: int, bottleneck: int = 128):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(in_dim, bottleneck), nn.Tanh(), nn.Linear(bottleneck, 1))
        self.out_dim = in_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)
        w = F.softmax(scores, dim=1)
        return torch.sum(x * w.unsqueeze(-1), dim=1), w


class TemporalEncoder(nn.Module):
    """Configurable speech temporal modelling block: gru | lstm | transformer | cnn | none."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, arch: str = "gru", num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.arch = arch.lower()
        if self.arch == "gru":
            self.net = nn.GRU(in_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                              bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
            self.out_dim = hidden_dim * 2
        elif self.arch == "lstm":
            self.net = nn.LSTM(in_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                               bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
            self.out_dim = hidden_dim * 2
        elif self.arch == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=in_dim, nhead=4, dim_feedforward=in_dim * 2,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.net = nn.TransformerEncoder(layer, num_layers=max(1, num_layers))
            self.out_dim = in_dim
        elif self.arch == "cnn":
            self.net = nn.Sequential(
                nn.Conv1d(in_dim, hidden_dim * 2, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.GELU(),
            )
            self.out_dim = hidden_dim * 2
        elif self.arch == "none":
            self.net = nn.Identity()
            self.out_dim = in_dim
        else:
            raise ValueError(f"Unknown temporal arch: {arch}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        if self.arch in ("gru", "lstm"):
            if mask is not None:
                lengths = mask.sum(dim=1).clamp(min=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
                out, _ = self.net(packed)
                out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
                return out
            out, _ = self.net(x)
            return out
        if self.arch == "transformer":
            pad_mask = ~mask if mask is not None else None
            return self.net(x, src_key_padding_mask=pad_mask)
        if self.arch == "cnn":
            return self.net(x.transpose(1, 2)).transpose(1, 2)
        return x


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
