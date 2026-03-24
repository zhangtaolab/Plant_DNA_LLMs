# GenomicRegLM 快速开始（当前可运行骨架）

## 1. 运行合成数据 smoke training

```bash
PYTHONPATH=src python -m genomicreglm.cli.train \
  --data-config configs/data/maize_multiomics.yaml \
  --model-config configs/model/transformer_multimodal.yaml \
  --train-config configs/train/finetune_expression_regression.yaml \
  --synthetic \
  --num-samples 64
```

## 2. 当前状态

- 已打通：配置读取 → tokenizer/model 构建 → dataset/dataloader → 训练循环。
- 暂未打通：真实 ATAC/ChIP/bigWig 数据读取（下一步补 `real data builder`）。
