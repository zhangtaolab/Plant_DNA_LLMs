from __future__ import annotations

from typing import Dict, List

import torch

from genomicreglm.tokenizers.simple_tokenizer import DNATokenizer


class MultimodalBatchCollator:
    def __init__(self, tokenizer: DNATokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([self.tokenizer.encode(item["sequence"]) for item in batch], dtype=torch.long)

        atac = torch.stack([item["atac_signal"] for item in batch], dim=0)
        chip = torch.stack([item["chip_signal"] for item in batch], dim=0)

        atac_feat = atac.mean(dim=1, keepdim=False)
        chip_feat = chip.mean(dim=2).mean(dim=1, keepdim=False)
        signal_features = torch.cat([atac_feat.unsqueeze(-1), chip_feat.unsqueeze(-1)], dim=1)

        labels = torch.stack([item["expression_target"] for item in batch], dim=0)

        return {
            "input_ids": input_ids,
            "signal_features": signal_features,
            "labels": labels,
        }
