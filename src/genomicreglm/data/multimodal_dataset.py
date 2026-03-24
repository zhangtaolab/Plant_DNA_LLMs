from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


@dataclass
class GeneSample:
    gene_id: str
    chrom: str
    start: int
    end: int
    strand: str
    tss: int
    sequence: str
    atac_signal: Sequence[float]
    chip_signal: Sequence[Sequence[float]]
    expression_target: float
    species: Optional[str] = None
    tissue: Optional[str] = None
    condition: Optional[str] = None


class MultimodalExpressionDataset(Dataset):
    """Minimal multimodal dataset interface for DNA + chromatin expression tasks.

    This class intentionally keeps the API simple so the team can plug in
    Arrow/HDF5 backends later without changing trainer/model contracts.
    """

    def __init__(self, samples: List[GeneSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        return {
            "gene_id": sample.gene_id,
            "sequence": sample.sequence,
            "atac_signal": torch.tensor(sample.atac_signal, dtype=torch.float32),
            "chip_signal": torch.tensor(sample.chip_signal, dtype=torch.float32),
            "expression_target": torch.tensor(sample.expression_target, dtype=torch.float32),
        }
