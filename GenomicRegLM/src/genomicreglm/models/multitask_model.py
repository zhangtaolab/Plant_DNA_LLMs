from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class MultitaskModelConfig:
    vocab_size: int = 8192
    d_model: int = 256
    signal_dim: int = 128
    n_targets: int = 1


class MultitaskExpressionModel(nn.Module):
    """Reference multimodal model skeleton.

    Current design:
    - sequence encoder: embedding + pooling (placeholder)
    - signal encoder: MLP on ATAC/ChIP summary features (placeholder)
    - fusion: concatenation + MLP
    - heads: regression head (required), classification head (optional)
    """

    def __init__(self, config: MultitaskModelConfig):
        super().__init__()
        self.config = config

        self.seq_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.signal_projector = nn.Sequential(
            nn.Linear(config.signal_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        self.fusion = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        self.regression_head = nn.Linear(config.d_model, config.n_targets)
        self.classification_head = nn.Linear(config.d_model, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        signal_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        seq_emb = self.seq_embedding(input_ids).mean(dim=1)
        sig_emb = self.signal_projector(signal_features)
        fused = self.fusion(torch.cat([seq_emb, sig_emb], dim=-1))

        reg_pred = self.regression_head(fused).squeeze(-1)
        cls_logits = self.classification_head(fused)

        outputs: Dict[str, torch.Tensor] = {
            "regression": reg_pred,
            "classification_logits": cls_logits,
        }
        if labels is not None:
            loss = nn.functional.mse_loss(reg_pred, labels)
            outputs["loss"] = loss

        return outputs
