# 环境配置
[参考](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/docs/zh/pytorch/training/install_guide.md)
## 激活CANN环境变量
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 
```
## 安装torch以及torch_npu
```bash
# 下载并安装PyTorch框架
wget https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl
pip3 install torch-2.7.1+cpu-cp310-cp310-manylinux_2_28_aarch64.whl
# 下载并安装torch_npu插件
wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.7.1/torch_npu-2.7.1.post2-cp310-cp310-manylinux_2_28_aarch64.whl
pip3 install torch_npu-2.7.1.post2-cp310-cp310-manylinux_2_28_aarch64.whl
```
## 安装MindSpeed加速库
```bash
git clone https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed
git checkout master  # checkout commit from MindSpeed master
pip3 install -r requirements.txt 
pip3 install -e .
cd ..
```
## 准备MindSpeed LLM及Megatron-LM源码
```
git clone https://gitcode.com/ascend/MindSpeed-LLM.git
git clone https://github.com/NVIDIA/Megatron-LM.git  # 从github下载Megatron-LM，请确保网络能访问
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-LLM/
cd ../MindSpeed-LLM
git checkout master
mkdir logs

pip3 install -r requirements.txt  # 安装其余依赖库
```
## 包冲突解决
numpy和scipy可能会与torch_npu存在版本冲突
```bash
 pip install numpy==1.24.3 scipy==1.10.1 --force-reinstall -i  https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

# 训练
## 预训练权重迁移
将在NVIDIA卡上训练完成的预训练权重文件(safetensor)转为到MindSpore的权重格式。

[参考](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/mamba2/ckpt_convert_mamba2_2.7b_mg2hf.sh)

**需要根据模型大小修改相关参数**
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CUDA_DEVICE_MAX_CONNECTIONS=1

python MindSpeed-LLM/convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --load-dir ./mamba2_130M_spv3 \
    --save-dir ./mamba2_130M_spv3_mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --hidden-size 768 \
    --mamba-state-dim 128 \
    --mamba-head-dim 64 \
    --mamba-num-groups 1 \
    --model-type-hf 'mamba2'
```

## 从头预训练
### 数据预处理
使用MindSpeed-LLM自带的数据预处理脚本，将预训练数据从TXT转为训练代码需要的数据格式parquet
```bash
python ./preprocess_data.py \
    --input datasets/spv3_sample.parquet \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ../mamba2_130M_spv3 \
    --output-prefix ./datasets/spv3_sample \
    --workers 8 \
    --seq-length 512 
```
## 预训练
使用shell脚本启动MindSpeed-LLM的预训练脚本

[参考](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/mamba2/pretrain_mamba2_hybrid_8b_4k_ptd.sh)

**脚本中部分参数需要根据模型大小和训练参数调整**
```bash
bash pretrain_mamba2_130M.sh
```

## 微调
在megatron/core/models/mamba/mamba_model.py中增加Mamba2ForSequenceClassification模型

[mamba_model.py](megatron_changed_file/mamba_model.py)

修改megatron/core/models/mamba/init.py便于后续调用

[init.py](megatron_changed_file/init.py)

微调脚本

[finetune_mamba2.py](finetune_mamba2.py)

启动微调脚本

**脚本中部分参数需要根据模型大小和训练参数调整**

[finetune_mamba2.sh](finetune_mamba2.sh)

启动微调
```bash
bash finetune_mamba2.sh
```