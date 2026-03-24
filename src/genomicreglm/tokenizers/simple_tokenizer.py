from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DNATokenizer:
    mode: str = "single_base"  # single_base | kmer_6
    max_length: int = 1024

    def __post_init__(self) -> None:
        bases = ["A", "C", "G", "T", "N"]
        self.vocab: Dict[str, int] = {"[PAD]": 0, "[UNK]": 1}
        if self.mode == "single_base":
            for i, b in enumerate(bases, start=2):
                self.vocab[b] = i
        else:
            # lightweight dynamic vocab for kmer mode
            for i, b in enumerate(bases, start=2):
                self.vocab[b] = i

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + 4096  # reserve room for dynamic kmers

    def _tokenize_single_base(self, seq: str) -> List[str]:
        return list(seq.upper())

    def _tokenize_kmer(self, seq: str, k: int = 6) -> List[str]:
        seq = seq.upper()
        if len(seq) < k:
            return [seq]
        return [seq[i : i + k] for i in range(len(seq) - k + 1)]

    def encode(self, seq: str) -> List[int]:
        tokens = self._tokenize_single_base(seq) if self.mode == "single_base" else self._tokenize_kmer(seq, k=6)
        ids: List[int] = []
        for tok in tokens[: self.max_length]:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
            ids.append(self.vocab.get(tok, self.vocab["[UNK]"]))
        if len(ids) < self.max_length:
            ids.extend([self.vocab["[PAD]"]] * (self.max_length - len(ids)))
        return ids
