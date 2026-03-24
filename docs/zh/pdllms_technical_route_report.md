# PDLLMs 训练思路与技术路线报告（结合论文与代码库）

## 1. 项目定位与论文结论对齐

论文《PDLLMs: A group of tailored DNA large language models for analyzing plant genomes》对应的仓库实现强调“**一组面向植物基因组的定制化 DNA 大模型**”，并非单一架构，而是以多架构（如 DNABERT、DNAGPT、DNAGemma、DNAMamba）+ 多 tokenizer（BPE / k-mer / single-base）进行系统化建模。仓库 README 与文档均明确这一点。

从工程视角看，代码库把论文主张落地为三层：

1. **基础模型层（Foundation DNA LLMs）**：在植物参考基因组序列上做语言建模预训练；
2. **任务适配层（Fine-tuning Heads + Metrics）**：面向启动子、保守性、组蛋白修饰、开放染色质、lncRNA、启动子强度等任务微调；
3. **应用部署层（Inference/API/Web）**：支持本地推理、API 推理与容器化部署。

---

## 2. 训练思路拆解（论文思想 × 代码实现）

## 2.1 数据与序列建模思想

仓库文档指出预训练数据应来自植物基因组切分序列（示例建议长度小于 2000bp 的片段），在训练脚本中再根据 `model_max_length`（默认 512）截断/补齐进入模型。该策略的本质是：

- 在工程上将超长 DNA 序列切成可批处理单元；
- 在模型上通过固定窗口学习局部模式（motif、调控片段组合）与统计依赖；
- 在后续下游任务中通过同样 tokenization 方式保持分布一致性。

## 2.2 预训练目标设计：CLM / MLM 双模式兼容

`model_pretrain_from_scratch.py` 中通过 `--is_mlm` 在两类目标间切换：

- `AutoModelForCausalLM`：自回归语言建模（更贴近生成式或下一 token 预测）；
- `AutoModelForMaskedLM`：掩码重建（更贴近双向上下文表征）。

脚本统一使用 Hugging Face `Trainer + DataCollatorForLanguageModeling` 完成训练调度，这意味着论文中“组模型”的实现策略不是重写训练器，而是复用统一训练骨架，靠“模型/分词器/目标函数”组合出多个变体。

## 2.3 分词器策略：BPE 与 k-mer 并行探索

从模型与微调资源列表看，项目显式维护了 BPE、6mer、single-base（以及 Mamba 的 2~6mer）多路线；`model_finetune.py` 则通过 `--kmer` 将 DNA 字符串在线转成空格分隔 k-mer token 序列。

这对应了论文核心工程策略：

- **BPE**：通过子词合并捕获高频序列模式，词表紧凑；
- **k-mer**：具备更强生物可解释性（局部组合单元明确）；
- **single-base**：保持原子级碱基分辨率。

实质上，这是在“统计效率、可解释性、长度开销”三者之间做任务相关的折中。

## 2.4 微调策略：单脚本覆盖分类/多分类/回归/多标签

`model_finetune.py` 通过 `--train_task` 统一处理 `classification`、`multi-classification`、`regression`、`multi-label`，并自动选择：

- 标签维度 (`num_labels`)；
- 对应模型头（`AutoModelForSequenceClassification` / token classification 分支）；
- 对应评价指标函数（`pdllib.metrics` 中的 accuracy/F1/MCC/AUROC/R2/Spearman 等）。

这说明论文中的“多任务泛化”在代码层面是**参数化任务模板**，而不是为每个任务单独造 pipeline。

## 2.5 训练稳定性与可复现性

代码里可以看到较完整的训练控制项：

- 随机种子固定；
- `warmup_ratio`、`weight_decay`、`gradient_accumulation_steps`；
- `fp16/bf16` 精度开关；
- checkpoint 与 best-model 机制。

这套设计支持从消费级单卡到多卡环境的迁移（文档里还给出 NCCL 相关开关提示），符合论文工具链“可用性优先”的定位。

---

## 3. 技术路线（可执行版本）

以下是基于论文方法与仓库能力抽象出的“从 0 到 1”落地路线。

## 阶段 A：目标定义与任务分层

1. 明确任务类型：分类/回归/多标签；
2. 定义输入序列长度窗口（如 256/512/1024）与标签体系；
3. 先建立统一评估口径（F1、MCC、AUROC、R2、Spearman）。

## 阶段 B：数据工程

1. 收集目标植物（或多物种）参考基因组；
2. 切分序列并清洗非法字符（N 比例阈值可配置）；
3. 形成：
   - 预训练文本（每行一段 DNA）；
   - 微调 CSV（`sequence,label`）；
4. 划分 train/dev/test，确保跨物种或染色体分层抽样，防止信息泄漏。

## 阶段 C：基础模型预训练（多路线并行）

建议并行训练至少 3 条主线：

- 路线 C1：DNAGPT + BPE（CLM）；
- 路线 C2：DNABERT + 6mer（MLM）；
- 路线 C3：DNAMamba + BPE 或 3/6mer（CLM）。

统一做法：

- 固定 `model_max_length` 与数据切片策略；
- 统一学习率/epoch/warmup 网格；
- 记录吞吐、loss 曲线与收敛速度。

## 阶段 D：任务微调与模型选择

1. 使用同一 `model_finetune.py`，只替换模型路径、任务类型与标签；
2. 对每个下游任务跑多架构 + 多 tokenizer；
3. 输出按任务维度的 leaderboard：性能、推理时延、显存占用、模型体积；
4. 选“任务最优模型”而非“单一全局最优”。

## 阶段 E：部署与应用

1. 用 `model_inference.py`/`api_inference.py` 打通本地与服务端推理；
2. 对外提供标准化输入（FASTA/CSV）与批处理 API；
3. 容器化（CPU/GPU docker）保障可迁移部署；
4. 建立模型版本管理（预训练版本 + 下游版本一一对应）。

---

## 4. 给你的项目实施建议（重点）

1. **先做小闭环，再做大一统**：
   - 先选 1 个高价值任务（如核心启动子识别）完成全链路；
   - 再横向扩展到 H3K27ac/开放染色质等任务。

2. **tokenizer 不要先入为主**：
   - 若追求可解释 motif，优先 6mer；
   - 若追求压缩效率与泛化，优先 BPE；
   - 最终以验证集结果与推理成本共同决策。

3. **把“多模型组”当成产品策略**：
   - 论文强调的是 tailored group；
   - 你的平台应按任务自动路由到最佳子模型，而不是强行 one-model-fits-all。

4. **评估要加跨物种泛化**：
   - 不仅看随机切分指标；
   - 建议增加 leave-one-species-out 评测，检验真实迁移能力。

---

## 5. 风险与改进方向

- **长程依赖受限**：默认 512 token 窗口可能不足以覆盖更长调控依赖；可尝试长上下文架构或分块聚合。
- **标签噪声**：组学标签（如染色质开放）实验噪声较大，建议引入置信度加权训练。
- **跨平台差异**：ModelScope/HuggingFace 权重与 tokenizer 版本要锁定哈希，避免“同名不同参”。
- **指标单一风险**：建议在 MCC/F1 之外加入校准指标（ECE/Brier）评估可用性。

---

## 6. 一句话总结

PDLLMs 的核心并不是“训练一个更大的 DNA 模型”，而是“围绕植物基因组场景，构建多架构 × 多 tokenization × 多任务的可选择模型组”，代码库已经把这条路线工程化为可复用的预训练与微调流水线；你可以直接在此基础上做任务化选型与平台化部署。
