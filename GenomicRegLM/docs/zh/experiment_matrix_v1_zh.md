# GenomicRegLM 第一版实验矩阵（V1）

本文件对应 `configs/experiment/exp_matrix_v1.yaml`，用于执行第一轮模型路线筛选。

## 1. 对比维度

- Backbone：Transformer vs Mamba
- Tokenizer：BPE vs 6-mer
- Fusion：Cross-attention vs Gated
- Split：染色体切分 / 跨组织 / 跨物种

## 2. 必报指标

- 回归：Pearson、Spearman、MSE、MAE
- 资源：训练吞吐（samples/s）、峰值显存、推理延迟

## 3. 结论规则

1. 先用染色体切分结果筛除明显弱模型；
2. 再看跨组织和跨物种泛化；
3. 若性能接近，优先训练成本更低与推理更快的方案。
