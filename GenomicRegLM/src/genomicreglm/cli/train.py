from __future__ import annotations

import argparse
import random

import torch
from torch.utils.data import DataLoader

from genomicreglm.config import load_yaml_config
from genomicreglm.data.collators import MultimodalBatchCollator
from genomicreglm.data.multimodal_dataset import MultimodalExpressionDataset
from genomicreglm.data.synthetic import build_synthetic_samples
from genomicreglm.factory import build_model, build_tokenizer
from genomicreglm.trainers.finetune_trainer import FinetuneConfig, FinetuneTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("GenomicRegLM training entrypoint")
    parser.add_argument("--data-config", type=str, required=True, help="Path to data yaml config")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model yaml config")
    parser.add_argument("--train-config", type=str, required=True, help="Path to train yaml config")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for smoke training")
    parser.add_argument("--num-samples", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    data_cfg = load_yaml_config(args.data_config)
    model_cfg = load_yaml_config(args.model_config)
    train_cfg = load_yaml_config(args.train_config)

    seed = int(train_cfg.get("seed", 7))
    random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = build_tokenizer(model_cfg)
    model = build_model(model_cfg, tokenizer)

    if args.synthetic:
        samples = build_synthetic_samples(n=args.num_samples)
        dataset = MultimodalExpressionDataset(samples)
    else:
        raise NotImplementedError(
            "Real data builder is not wired yet. Use --synthetic for now."
        )

    collator = MultimodalBatchCollator(tokenizer)
    batch_size = int(train_cfg.get("batch_size", 8))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    trainer_cfg = FinetuneConfig(
        epochs=int(train_cfg.get("epochs", 5)),
        lr=float(train_cfg.get("optimizer", {}).get("lr", 1e-4)),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    trainer = FinetuneTrainer(model, trainer_cfg)
    trainer.fit(dataloader)

    print(
        f"[GenomicRegLM] done | data={data_cfg.get('name', 'unknown')} | model={model_cfg.get('name', 'unknown')}"
    )


if __name__ == "__main__":
    main()
