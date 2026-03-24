from __future__ import annotations

import random
from typing import List

from genomicreglm.data.multimodal_dataset import GeneSample


def random_dna(length: int) -> str:
    return "".join(random.choice("ACGT") for _ in range(length))


def build_synthetic_samples(n: int = 64, seq_len: int = 256, signal_len: int = 128, n_marks: int = 2) -> List[GeneSample]:
    samples: List[GeneSample] = []
    for i in range(n):
        seq = random_dna(seq_len)
        atac = [random.random() for _ in range(signal_len)]
        chip = [[random.random() for _ in range(signal_len)] for _ in range(n_marks)]

        # simple pseudo-label rule for smoke training
        gc_ratio = (seq.count("G") + seq.count("C")) / max(1, len(seq))
        target = 0.4 * gc_ratio + 0.3 * (sum(atac) / signal_len) + 0.3 * (sum(sum(x) for x in chip) / (signal_len * n_marks))

        samples.append(
            GeneSample(
                gene_id=f"gene_{i}",
                chrom="chr1",
                start=i * 1000,
                end=i * 1000 + 500,
                strand="+",
                tss=i * 1000 + 100,
                sequence=seq,
                atac_signal=atac,
                chip_signal=chip,
                expression_target=target,
                species="synthetic",
                tissue="leaf",
                condition="control",
            )
        )
    return samples
