"""Multimodal Speech+Text Emotion Recognition models.

Supports:
  - concat:       early feature concatenation [h_s; h_t] -> MLP
  - late:         learnable weighted average of speech & text logits
  - gated_xattn:  bidirectional cross-modal attention + GMU-style scalar/vector gate z in [0,1]
  - align_pool:   alignment-aware fusion (pools speech frames separately over carrier vs target-word
                  spans derived from MFA forced alignment, fused with text & gated attention)
"""
import torch
import torch.nn as nn

from mer.models.blocks import AttentiveStatsPool, AttnPool, MLPHead
from mer.models.speech import SpeechBranch
from mer.models.text import TextBranch


class MultimodalEmotionModel(nn.Module):
    def __init__(
        self,
        speech_cfg: dict,
        text_cfg: dict,
        fusion_type: str = "gated_xattn",
        fusion_dim: int = 256,
        num_heads: int = 4,
        num_classes: int = 7,
        dropout: float = 0.3,
        modality_dropout: float = 0.2,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.modality_dropout = modality_dropout

        self.speech = SpeechBranch(dropout=dropout, **speech_cfg)
        self.text = TextBranch(dropout=dropout, **text_cfg)

        self.s_proj = nn.Sequential(nn.Linear(self.speech.out_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.t_proj = nn.Sequential(nn.Linear(self.text.out_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())

        if fusion_type == "concat":
            self.fuse_proj = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
            self.head = MLPHead(fusion_dim, fusion_dim, num_classes, dropout)
        elif fusion_type == "late":
            self.s_head = MLPHead(fusion_dim, fusion_dim, num_classes, dropout)
            self.t_head = MLPHead(fusion_dim, fusion_dim, num_classes, dropout)
            self.alpha = nn.Parameter(torch.tensor(0.5))
        elif fusion_type == "gated_xattn":
            self.s_frame_proj = nn.Linear(self.speech.frame_dim, fusion_dim)
            self.t_tok_proj = nn.Linear(self.text.frame_dim, fusion_dim)
            self.s2t_attn = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
            self.t2s_attn = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
            self.s_norm = nn.LayerNorm(fusion_dim)
            self.t_norm = nn.LayerNorm(fusion_dim)
            self.s_xpool = AttnPool(fusion_dim)
            self.t_xpool = AttnPool(fusion_dim)
            self.gate = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Sigmoid())
            self.head = MLPHead(fusion_dim, fusion_dim, num_classes, dropout)
        elif fusion_type == "align_pool":
            # Pool speech separately over target-word frames and carrier-phrase frames (from MFA)
            self.target_pool = AttentiveStatsPool(self.speech.frame_dim)
            self.carrier_pool = AttentiveStatsPool(self.speech.frame_dim)
            self.span_proj = nn.Sequential(
                nn.Linear(self.speech.frame_dim * 4, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
            )
            self.gate = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Sigmoid())
            self.head = MLPHead(fusion_dim, fusion_dim, num_classes, dropout)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def _apply_mod_dropout(self, h_s: torch.Tensor, h_t: torch.Tensor):
        if not self.training or self.modality_dropout <= 0:
            return h_s, h_t
        B = h_s.size(0)
        # Independently drop speech or text with prob p (never both simultaneously)
        r = torch.rand(B, device=h_s.device)
        drop_t = (r < self.modality_dropout).float().unsqueeze(-1)
        drop_s = ((r >= self.modality_dropout) & (r < 2 * self.modality_dropout)).float().unsqueeze(-1)
        return h_s * (1.0 - drop_s), h_t * (1.0 - drop_t)

    def forward(
        self,
        ssl_feats: torch.Tensor,
        speech_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        zero_text: bool = False,
        zero_speech: bool = False,
    ):
        s_out = self.speech(ssl_feats, speech_mask)
        t_out = self.text(input_ids, attention_mask)

        h_s = self.s_proj(s_out["emb"])
        h_t = self.t_proj(t_out["emb"])

        if zero_text:
            h_t = torch.zeros_like(h_t)
        if zero_speech:
            h_s = torch.zeros_like(h_s)

        gate_val = None
        if self.fusion_type == "concat":
            h_s_d, h_t_d = self._apply_mod_dropout(h_s, h_t)
            fused = self.fuse_proj(torch.cat([h_s_d, h_t_d], dim=-1))
            logits = self.head(fused)
        elif self.fusion_type == "late":
            w = torch.sigmoid(self.alpha)
            logits = w * self.s_head(h_s) + (1.0 - w) * self.t_head(h_t)
            fused = w * h_s + (1.0 - w) * h_t
            gate_val = w.expand(h_s.size(0), 1)
        elif self.fusion_type == "gated_xattn":
            s_seq = self.s_frame_proj(s_out["frames"])
            t_seq = self.t_tok_proj(t_out["tokens"])
            if zero_text:
                t_seq = torch.zeros_like(t_seq)
            if zero_speech:
                s_seq = torch.zeros_like(s_seq)
            s_pad = ~speech_mask
            t_pad = ~attention_mask.bool()
            s_att, _ = self.s2t_attn(s_seq, t_seq, t_seq, key_padding_mask=t_pad)
            t_att, _ = self.t2s_attn(t_seq, s_seq, s_seq, key_padding_mask=s_pad)
            s_ctx = self.s_norm(s_seq + s_att)
            t_ctx = self.t_norm(t_seq + t_att)
            hs_x, _ = self.s_xpool(s_ctx, speech_mask)
            ht_x, _ = self.t_xpool(t_ctx, attention_mask.bool())
            hs_x, ht_x = self._apply_mod_dropout(hs_x, ht_x)
            z = self.gate(torch.cat([hs_x, ht_x], dim=-1))  # [B, D], higher = more speech
            fused = z * hs_x + (1.0 - z) * ht_x
            logits = self.head(fused)
            gate_val = z.mean(dim=-1, keepdim=True)  # scalar summary per sample
        elif self.fusion_type == "align_pool":
            frames = s_out["frames"]
            if target_mask is None:
                # Fallback if no alignment mask: second half of valid frames
                target_mask = speech_mask.clone()
            t_m = target_mask & speech_mask
            # Guard empty mask
            empty_t = t_m.sum(dim=1) == 0
            if empty_t.any():
                t_m = t_m.clone()
                t_m[empty_t] = speech_mask[empty_t]
            c_m = (~t_m) & speech_mask
            empty_c = c_m.sum(dim=1) == 0
            if empty_c.any():
                c_m = c_m.clone()
                c_m[empty_c] = speech_mask[empty_c]
            h_target, _ = self.target_pool(frames, t_m)
            h_carrier, _ = self.carrier_pool(frames, c_m)
            h_span = self.span_proj(torch.cat([h_target, h_carrier], dim=-1))
            if zero_speech:
                h_span = torch.zeros_like(h_span)
            h_span_d, h_t_d = self._apply_mod_dropout(h_span, h_t)
            z = self.gate(torch.cat([h_span_d, h_t_d], dim=-1))
            fused = z * h_span_d + (1.0 - z) * h_t_d
            logits = self.head(fused)
            gate_val = z.mean(dim=-1, keepdim=True)

        return {
            "logits": logits,
            "fusion_emb": fused,
            "temporal_emb": s_out["emb"],
            "contextual_emb": t_out["emb"],
            "gate": gate_val,
        }
