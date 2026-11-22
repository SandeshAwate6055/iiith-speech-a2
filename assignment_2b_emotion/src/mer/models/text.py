"""Text-only Emotion Recognition model (T*): RoBERTa contextual encoder -> attention pooling -> MLP."""
import torch
import torch.nn as nn
from transformers import AutoModel

from mer.models.blocks import AttnPool, MLPHead


class TextBranch(nn.Module):
    def __init__(
        self,
        model_name: str = "roberta-base",
        n_finetune_layers: int = 2,
        proj_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        # Freeze all parameters first, then unfreeze top-k transformer layers
        for p in self.encoder.parameters():
            p.requires_grad = False
        if n_finetune_layers > 0:
            layers = None
            if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
                layers = self.encoder.encoder.layer
            elif hasattr(self.encoder, "transformer") and hasattr(self.encoder.transformer, "layer"):
                layers = self.encoder.transformer.layer
            if layers is not None:
                for layer in layers[-n_finetune_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )
        self.pool = AttnPool(proj_dim)
        self.frame_dim = proj_dim
        self.out_dim = proj_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        tokens = self.proj(out.last_hidden_state)  # [B, N, proj_dim]
        mask = attention_mask.bool()
        emb, attn_w = self.pool(tokens, mask)
        return {"tokens": tokens, "emb": emb, "attn_weights": attn_w}


class TextEmotionModel(nn.Module):
    def __init__(self, num_classes: int = 7, head_hidden: int = 256, dropout: float = 0.3, **branch_kwargs):
        super().__init__()
        self.branch = TextBranch(dropout=dropout, **branch_kwargs)
        self.head = MLPHead(self.branch.out_dim, head_hidden, num_classes, dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_):
        out = self.branch(input_ids, attention_mask)
        logits = self.head(out["emb"])
        return {
            "logits": logits,
            "contextual_emb": out["emb"],
            "attn_weights": out["attn_weights"],
        }
