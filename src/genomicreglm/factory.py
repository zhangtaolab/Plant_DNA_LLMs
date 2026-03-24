from __future__ import annotations

from typing import Any, Dict

from genomicreglm.models.multitask_model import MultitaskExpressionModel, MultitaskModelConfig
from genomicreglm.tokenizers.simple_tokenizer import DNATokenizer


def build_tokenizer(model_cfg: Dict[str, Any]) -> DNATokenizer:
    seq_cfg = model_cfg.get("sequence_encoder", {})
    tok = seq_cfg.get("tokenizer", "single_base")
    mode = "kmer_6" if "kmer" in str(tok) else "single_base"
    max_len = int(seq_cfg.get("max_length", 1024))
    return DNATokenizer(mode=mode, max_length=max_len)


def build_model(model_cfg: Dict[str, Any], tokenizer: DNATokenizer) -> MultitaskExpressionModel:
    seq_cfg = model_cfg.get("sequence_encoder", {})
    sig_cfg = model_cfg.get("signal_encoder", {})
    config = MultitaskModelConfig(
        vocab_size=max(tokenizer.vocab_size, int(seq_cfg.get("vocab_size", 8192))),
        d_model=int(seq_cfg.get("d_model", 256)),
        signal_dim=2,
        n_targets=1,
    )
    return MultitaskExpressionModel(config)
