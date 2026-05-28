#!/bin/bash

# ================= 环境变量优化 =================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600
export TASK_QUEUE_ENABLE=2

# ================= 硬件与分布式配置 (单卡模式) =================
NPUS_PER_NODE=1        # 修改为 1 卡
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=1           # 总卡数为 1

# ================= 路径配置 (请确保这些路径正确) =================
# 建议使用绝对路径
CKPT_SAVE_DIR="/root/autodl-tmp/mamba2_130M_spv3_pretrained"
DATA_PATH="/root/autodl-tmp/MindSpeed-LLM/datasets/spv3_sample_text_document"
TOKENIZER_PATH="/root/autodl-tmp/dnamamba2_tokenizer"

# ================= 训练超参数 (单卡适配) =================
TP=1
PP=1
MBS=32                 # 130M模型较小，单卡32比较稳妥
GBS=32                 # 单卡时 GBS 必须等于 MBS (除非开启梯度累加)

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# ================= Mamba2 130M 核心架构参数 =================
MAMBA_ARGS="
    --spec mindspeed_llm.tasks.models.spec.mamba_spec layer_spec \
    --reuse-fp32-param \
    --use-distributed-optimizer \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 24 \
    --hidden-size 768 \
    --num-attention-heads 24 \
    --num-query-groups 24 \
    --mamba-num-groups 1 \
    --mamba-chunk-size 256 \
    --mamba-state-dim 128 \
    --mamba-d-conv 4 \
    --mamba-expand 2 \
    --mamba-head-dim 64 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --position-embedding-type none \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --train-iters 156250 \
    --lr 6e-4 \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --lr-warmup-iters 100 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0 \
    --initial-loss-scale 65536 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 98,2,0 \
    --workers 4
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 5000 \
    --eval-interval 5000 \
    --eval-iters 625 \
    --save ${CKPT_SAVE_DIR} \
    --no-save-rng \
    --no-save-optim
"

CKPT_ARGS="
    --model-type-hf mamba2
"

# ================= 启动命令 =================
mkdir -p logs
mkdir -p ${CKPT_SAVE_DIR}

python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_mamba.py \
    $MAMBA_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $CKPT_ARGS \
    --distributed-backend nccl \
    --transformer-impl local \
    | tee logs/pretrain_mamba2_130M_single_card.log