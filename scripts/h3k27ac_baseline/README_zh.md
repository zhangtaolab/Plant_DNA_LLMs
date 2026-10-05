# H3K27ac CPU 基线与复现实验

复用仓库 `data/train.csv`、`dev.csv`、`test.csv` 中的 40,000 / 10,000 / 10,000 条序列，不下载新数据。
提供序列组成基线和反向互补不变的 canonical 5-mer TF-IDF + 正则化逻辑回归。
这是 H3K27ac 二分类工具，不是 GenomicRegLM 的真实基因表达预测模型，也不是 DNA 基础大模型。

## 哪些文件必须保留

| 文件 | 用途 | 是否必须 |
|---|---|---|
| `features.py`、`predict.py`、`requirements.txt` | 特征提取、CPU 预测和精简依赖 | 直接预测必须 |
| `artifacts/canonical_5mer_model.joblib` | 已训练模型，约 13 KB | 直接预测必须；重新训练时可生成 |
| `benchmark.py` 和仓库现有 `data/*.csv` | 重新训练、验证集选参和测试集评估 | 复现实验必须 |
| `artifacts/report.json` | 数据 SHA256、参数、指标、限制与运行版本 | 结果审阅所需；运行时不读取 |
| `artifacts/SHA256SUMS` | 已提交模型及报告/预测的校验值 | 验证产物完整性时使用 |
| `artifacts/test_predictions.csv` | 本次测试的逐样本标签与两种模型概率 | 可选，用于复核指标；没有重复保存序列 |
| `environment/requirements-cpu.txt` | 整个仓库 CPU 环境的完整固定版本 | 可选；单独使用基线不需要 Torch |
| `environment/install_cpu.sh`、`activate.sh` | 重建及激活完整 CPU 环境 | 可选；运行 PDLLMs/GenomicRegLM 时使用 |
| `smoke_inference.py` | 检查 pdllib 本地模型加载、批量及 FASTA 推理 | 可选；不能评估生物学准确率 |

不提交虚拟环境、包缓存、安装日志、重复预测文件或随机测试模型。
`outputs/` 是忽略的派生结果目录；重新运行训练会替换其中的同名输出，不会修改原始数据或已提交的 `artifacts/`。

## 快速使用已训练模型

以下命令从仓库根目录运行，推荐 Python 3.11；使用已有虚拟环境或新建环境。

```bash
python -m pip install -r scripts/h3k27ac_baseline/requirements.txt
mkdir -p scripts/h3k27ac_baseline/outputs
python scripts/h3k27ac_baseline/predict.py \
  --input data/test.csv \
  --output scripts/h3k27ac_baseline/outputs/predictions.csv
```

输入 CSV 必须含 `sequence` 列，序列会转换为大写，允许 A/C/G/T/N，至少 5 bp。
输出包含行号、H3K27ac 概率和使用验证集阈值 0.51 的分类。
为了避免覆盖文件，预测输出路径必须尚不存在。概率未经校准，模型在当前划分上的验证不能保证跨物种或其他实验条件下的适用性。
`joblib` 模型使用 pickle 反序列化，只加载本仓库发布或自己训练的可信文件；不要加载不可信来源的模型文件。

## 复现实验

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python scripts/h3k27ac_baseline/benchmark.py \
  --data-dir data \
  --output-dir scripts/h3k27ac_baseline/outputs \
  --seed 7
```

输出 `report.json`、`canonical_5mer_model.joblib` 和 `test_predictions.csv`。
如使用重新训练的模型预测，通过 `--model scripts/h3k27ac_baseline/outputs/canonical_5mer_model.joblib` 指定。
默认数据和模型路径根据脚本位置解析，不依赖 `/workspace` 或某个固定检出目录。

固定词表覆盖所有合法 5-mer 的正反链规范形式；TF-IDF 仅在训练集拟合。
训练集优先于验证集、验证集优先于测试集去除精确或反向互补重复；当前数据没有此类重复或非法样本。
三个正则化参数 C = 0.1 / 1 / 10 和分类阈值仅在 dev 上选择，选定后评估 test。
输出审计记录、数据 SHA256 和各类指标；序列缺少染色体、物种和坐标元信息，因此不能验证染色体隔离或排除同源/部分重叠泄漏。

## 已验证的结果

| 测试集指标 | 序列组成基线 | canonical 5-mer |
|---|---:|---:|
| AUROC | 0.67395 | 0.77048 |
| AUPRC | 0.67770 | 0.77516 |
| MCC | 0.23614 | 0.41081 |
| F1 | 0.64308 | 0.69500 |
| Accuracy | 0.61670 | 0.70500 |

已在完整的现有划分上训练并评估，独立预测 CLI 已核对全部 10,000 条测试预测。
这是单种子、单划分上的经典模型比较，不是原 PDLLM 的改进证明。
测试集已经用于报告最终性能，不应继续根据它选模型；后续改进仍须使用验证集及独立的新测试集。

## 可选：完整 CPU 开发环境

已验证 Python 3.11.16、Torch 2.3.1+cpu、Transformers 4.45.2。
完整固定版本记录在 `environment/requirements-cpu.txt`；基线的精简依赖位于本目录 `requirements.txt`。
安装脚本要求已有 `uv`，默认创建仓库根目录下 `.venv`，也可传入环境的绝对路径。
若指定已有环境，其依赖会调整为固定版本；为保护其他工作，优先使用专用环境。

```bash
bash scripts/h3k27ac_baseline/environment/install_cpu.sh
source scripts/h3k27ac_baseline/environment/activate.sh
```

脚本保留 TLS 验证，先从官方 PyTorch CPU 索引安装固定 Torch，再安装完整依赖，初始化仓库固定的 evaluate 子模块。
随后运行本地随机小 BERT 的推理检查和 GenomicRegLM 的 64 样本、20 轮合成训练。
随机小 BERT 保存在自动清理的临时目录；无需下载预训练权重。完整开发环境不需要后台服务。
缓存默认位于仓库已有忽略目录 `.cache/`，已有环境变量会保留。每个云任务已经隔离，使用现有检出，不需要创建 Git worktree。

当前验证只覆盖 CPU，不包含 CUDA/Mamba 扩展、大规模预训练或真实组学数据读取。
GenomicRegLM 目前是嵌入池化和 MLP 骨架，合成数据训练不能代表真实表达预测性能。

## 文献和后续比较

参考作者官方实现的方法说明：

- [DNABERT-2](https://arxiv.org/abs/2306.15006) / [作者实现](https://github.com/MAGICS-LAB/DNABERT_2)：多物种预训练、BPE 和 ALiBi。本次未运行该基础模型。
- [Caduceus](https://arxiv.org/abs/2403.03234) / [作者实现](https://github.com/kuleshov-group/caduceus)：反向互补对称性及双向长序列建模。本工具仅借鉴链对称原则，不是 Caduceus 实现。
- [PDLLMs，Molecular Plant 2025](https://doi.org/10.1016/j.molp.2024.12.006)：作者为 H3K27ac 推荐 Plant DNAMamba 6mer；本次未取得其同划分成绩或运行其权重。

不能将不同物种、任务、序列长度和划分下的论文指标直接用于排名。
真实表达或变异效应预测需要匹配的表达/表型标签、基因组坐标和实验信息，不能以 H3K27ac 标签或拟南芥 1001 的基因组数据代替。
后续应先检查已有数据与缓存，只下载缺失部分，再实施带同源簇/染色体/物种隔离的验证和文献模型对照。
