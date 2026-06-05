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
CKPT_LOAD_DIR="/root/autodl-tmp/mamba2_130M_spv3"
CKPT_SAVE_DIR="/root/autodl-tmp/mamba2_130M_spv3_finetuned_V2"
TOKENIZER_PATH="/root/autodl-tmp/mamba2_130M_spv3"
TRAIN_CSV="/root/autodl-tmp/datasets/train.csv"
DEV_CSV="/root/autodl-tmp/datasets/dev.csv"
TEST_CSV="/root/autodl-tmp/datasets/test.csv"



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
    --no-shared-storage \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 24 \
    --num_labels 3 \
    --problem_type single_label_classification \
    --group-query-attention \
    --num-query-groups 24 \
    --mamba-num-groups 1 \
    --mamba-chunk-size 256 \
    --mamba-state-dim 128 \
    --mamba-d-conv 4 \
    --mamba-expand 2 \
    --mamba-head-dim 64 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --hidden-size 768 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --num-attention-heads 24 \
    --make-vocab-size-divisible-by 1 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type none \
    --normalization RMSNorm \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --lr 2.5e-5 \
    --min-lr 2.5e-6 \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 65536 \
    --rotary-base 10000 \
    --no-gradient-accumulation-fusion \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --no-save-rng \
    --no-save-optim
"

CKPT_ARGS="
    --enable-hf2mg-convert \
    --model-type-hf mamba2
"

EXTRA_ARGS="
    --train_csv $TRAIN_CSV \
    --dev_csv $DEV_CSV \
    --test_csv $TEST_CSV \
    --tensorboard_dir ${CKPT_SAVE_DIR}/tensorboard_logs
"

# ================= 启动命令 =================
mkdir -p logs
mkdir -p ${CKPT_SAVE_DIR}

python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_mamba2.py \
    $MAMBA_ARGS \
    $OUTPUT_ARGS \
    $CKPT_ARGS \
    $EXTRA_ARGS \
    --distributed-backend nccl \
    --transformer-impl local \
    | tee logs/finetune_mamba2_130M_single_card.log