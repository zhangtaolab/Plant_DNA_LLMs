# 全新代码库技术路线图（DNA序列 + 染色质状态 → 基因表达预测）

> 目标：构建一个“多模态基因调控基础模型”代码库，输入 DNA 序列、ATAC-seq、ChIP-seq、基因组坐标与样本条件，输出基因表达水平（回归）与表达分层（可选分类）。

---

## 1. 任务定义与设计原则

## 1.1 任务定义

- **主任务（回归）**：预测基因表达值（TPM/RPKM/FPKM/log1p(counts)）。
- **辅任务（多任务学习）**：
  - 表达上/下调分类；
  - 组织/条件特异表达分类；
  - 可解释性头：预测与表达相关的关键调控区域贡献分数。

## 1.2 输入模态

1. **DNA序列**：TSS 上下游窗口（例如 [-50kb, +10kb] 可配置）。
2. **ATAC-seq**：可及性信号轨道（bigWig / bedGraph）映射到同一窗口。
3. **ChIP-seq**：组蛋白修饰/TF结合信号（多通道）。
4. **基因组注释**：基因坐标、链方向、TSS/TES、增强子候选区域。
5. **样本元信息**：物种、组织、处理条件、批次等。

## 1.3 设计原则

- **多模态统一表示**：序列 token 与表观信号 patch 在统一坐标系中对齐。
- **可插拔骨干网络**：Transformer / Mamba / Hybrid 可切换。
- **任务头解耦**：表达回归头与辅助任务头解耦，便于扩展。
- **评估可追溯**：跨物种、跨组织、跨平台三类泛化评估必须独立报告。

---

## 2. 新代码库架构（建议目录）

```text
GenomicRegLM/
├─ README.md
├─ pyproject.toml
├─ configs/
│  ├─ data/
│  │  ├─ maize_multiomics.yaml
│  │  └─ arabidopsis_multiomics.yaml
│  ├─ model/
│  │  ├─ transformer_multimodal.yaml
│  │  ├─ mamba_multimodal.yaml
│  │  └─ hybrid_xattn.yaml
│  ├─ train/
│  │  ├─ pretrain_masked_multimodal.yaml
│  │  ├─ finetune_expression_regression.yaml
│  │  └─ multitask_expression.yaml
│  └─ experiment/
│     ├─ cv_cross_species.yaml
│     └─ leave_tissue_out.yaml
├─ src/
│  └─ genomicreglm/
│     ├─ data/
│     │  ├─ genome_index.py
│     │  ├─ sequence_dataset.py
│     │  ├─ signal_dataset.py
│     │  ├─ multimodal_dataset.py
│     │  ├─ collators.py
│     │  └─ interval_sampler.py
│     ├─ tokenizers/
│     │  ├─ bpe_tokenizer.py
│     │  ├─ kmer_tokenizer.py
│     │  ├─ byte_tokenizer.py
│     │  └─ tokenizer_router.py
│     ├─ models/
│     │  ├─ encoders/
│     │  │  ├─ dna_transformer.py
│     │  │  ├─ dna_mamba.py
│     │  │  └─ signal_cnn.py
│     │  ├─ fusion/
│     │  │  ├─ cross_attention_fusion.py
│     │  │  ├─ gated_fusion.py
│     │  │  └─ coordinate_positional_encoding.py
│     │  ├─ heads/
│     │  │  ├─ expression_regression_head.py
│     │  │  ├─ expression_classification_head.py
│     │  │  └─ uncertainty_head.py
│     │  └─ multitask_model.py
│     ├─ losses/
│     │  ├─ weighted_mse.py
│     │  ├─ pearson_loss.py
│     │  └─ multitask_loss.py
│     ├─ metrics/
│     │  ├─ regression.py
│     │  ├─ ranking.py
│     │  └─ calibration.py
│     ├─ trainers/
│     │  ├─ pretrain_trainer.py
│     │  ├─ finetune_trainer.py
│     │  └─ distributed.py
│     ├─ inference/
│     │  ├─ predictor.py
│     │  ├─ batch_predict.py
│     │  └─ explain.py
│     └─ cli/
│        ├─ train.py
│        ├─ evaluate.py
│        └─ predict.py
├─ scripts/
│  ├─ build_multimodal_h5.py
│  ├─ run_pretrain.sh
│  ├─ run_finetune.sh
│  └─ run_benchmark.sh
├─ tests/
│  ├─ test_data_pipeline.py
│  ├─ test_model_forward.py
│  ├─ test_loss_metric.py
│  └─ test_cli.py
└─ docs/
   ├─ architecture.md
   ├─ data_schema.md
   ├─ training_protocol.md
   └─ benchmark_protocol.md
```

---

## 3. 数据建模与样本构建（核心）

## 3.1 统一样本单元

以“基因 × 样本（组织/处理）”为基本样本，构建如下字段：

- `gene_id`
- `chrom, start, end, strand, tss`
- `dna_seq_window`（围绕 TSS/TES 或 enhancer-gene link 的拼接窗口）
- `atac_signal_vector`（按 bin 聚合，如 50bp/100bp）
- `chip_signal_matrix`（mark × bins）
- `expression_target`
- `sample_meta`

## 3.2 对齐策略

- 坐标统一到同一参考基因组版本。
- ATAC/ChIP 统一做 binning + normalization（CPM/RPKM/quantile 可配置）。
- 负链基因进行方向标准化（序列反向互补 + 信号同步翻转）。

## 3.3 数据切分策略

必须同时提供三种切分：

1. **随机切分**（基线）；
2. **按染色体切分**（防止邻域泄漏）；
3. **跨组织/跨物种切分**（验证泛化）。

---

## 4. 模型路线（从可行到增强）

## 路线A：基线多模态Transformer

- DNA encoder：BPE/k-mer token + Transformer。
- Signal encoder：1D CNN/Transformer 处理 ATAC+ChIP bins。
- Fusion：Cross-Attention（序列 query，信号 key/value）。
- Head：回归输出表达值。

适合作为首个可用版本（V1）。

## 路线B：长序列Mamba增强

- DNA backbone 替换为 Mamba（更优长序列效率）。
- 信号通道保留 CNN 或轻量 Mamba。
- 融合层继续使用 gated fusion / cross-attention。

适合大窗口（>20kb）与高吞吐训练（V2）。

## 路线C：Hybrid（Transformer + Mamba）

- 前层用 Mamba 建模长程依赖。
- 高层用 Transformer 做跨模态精细对齐。
- 增加不确定性估计头（Aleatoric/Ensemble）。

适合追求精度上限与可解释性（V3）。

---

## 5. 分词器策略（借鉴并扩展 PDLLMs 思路）

保留“多 tokenizer 并行评估”原则：

1. **Single-base**：最高分辨率，序列长；
2. **k-mer (3~6)**：解释性好，适合 motif；
3. **BPE**：压缩效率高；
4. **混合策略**：
   - DNA主干用 BPE；
   - motif 辅助分支用 6-mer；
   - 融合时做 late-fusion。

结论不预设，以任务性能 + 显存成本 + 推理延迟共同决策。

---

## 6. 训练范式

## 6.1 预训练（可选但推荐）

在无标签多模态基因组区间上做自监督：

- DNA MLM/CLM；
- Signal masking reconstruction（遮蔽 ATAC/ChIP bins 复原）；
- 序列-信号对比学习（InfoNCE）。

目标是学习跨模态共享表征，减轻下游标注数据不足问题。

## 6.2 微调（主线）

- 主损失：`MSE` 或 `Huber`（表达回归）；
- 辅损失：Pearson/Spearman 导向损失（提升相关性）；
- 多任务损失加权：不确定性加权或动态权重平均（DWA）。

## 6.3 训练稳定性

- mixed precision（bf16）；
- gradient accumulation；
- warmup + cosine decay；
- early stopping 监控 `val_spearman`。

---

## 7. 评估协议（必须落实）

## 7.1 回归指标

- Pearson r
- Spearman ρ
- MSE / MAE
- 按表达分位数分层误差（低表达/高表达）

## 7.2 排序与实用指标

- Top-k 高表达基因召回率
- Differential expression ranking consistency

## 7.3 泛化评估

- Leave-one-chromosome-out
- Leave-one-tissue-out
- Leave-one-species-out

## 7.4 解释性评估

- in-silico mutagenesis 对表达变化敏感位点
- ATAC/ChIP 通道归因（Integrated Gradients / SHAP）
- 已知 motif 富集一致性检验

---

## 8. 工程落地计划（12周建议）

## Phase 0（第1-2周）：基础设施

- 建仓库、配置系统、数据 schema、单元测试框架。

## Phase 1（第3-5周）：数据管线

- 完成序列抽取、ATAC/ChIP对齐、表达标签构建。
- 输出可复用的 HDF5/Arrow 数据集。

## Phase 2（第6-8周）：V1基线模型

- 多模态 Transformer + BPE/6mer 两条 tokenizer 实验。
- 跑通随机/染色体切分基准。

## Phase 3（第9-10周）：Mamba增强

- 接入 Mamba backbone，比较长窗口收益与效率。

## Phase 4（第11-12周）：泛化与部署

- 完成跨组织/跨物种评估。
- 导出推理 API 与批量预测脚本。

---

## 9. 关键风险与预案

1. **多模态缺失**：部分基因缺失 ChIP mark
   - 预案：mask-aware fusion + 模态 dropout。
2. **批次效应强**：不同实验平台信号偏移
   - 预案：ComBat/对抗域适配/条件归一化。
3. **坐标与注释版本不一致**
   - 预案：强制 liftover + metadata 校验。
4. **高表达尾部难学**
   - 预案：log-transform + quantile loss。

---

## 10. 你可直接审核的交付件清单

1. 架构草图（目录级）
2. 数据 schema（字段定义 + 单位 + 坐标系）
3. 训练配置模板（pretrain/finetune）
4. 评估 protocol（含跨物种设置）
5. ablation 计划（tokenizer × backbone × fusion）

> 如果你认可该路线，下一步我可以按这个蓝图继续输出：
> - `configs/*.yaml` 的最小可运行模板
> - `src/` 下核心模块接口（Dataset/Model/Trainer）
> - 第一版实验矩阵（8~12个实验）
