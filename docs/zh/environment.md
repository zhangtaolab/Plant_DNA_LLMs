推荐使用[Anaconda](https://docs.anaconda.com/free/anaconda/install/)包管理工具构建模型训练环境。对于预训练和微调模型，请确保你的电脑包含NVIDIA显卡，并且对应的显卡驱动正确安装。对于模型的推理，不包含显卡的设备，如纯CPU、苹果芯片等也可以使用。

#### 1.1 下载并安装 [Anaconda](https://www.anaconda.com/download)

#### 1.2 创建运行环境 (本环境已在python 3.11中测试)

```bash
conda create -n llms python=3.11
conda activate llms
```

#### 1.3 安装依赖

模型训练和推理需要确保[NVIDIA显卡的驱动](https://www.nvidia.com/download/index.aspx)被正确安装，此外对应的[CUDA驱动](https://developer.nvidia.com/cuda-downloads)（版本>11.0, 本环境使用CUDA 12.1）也要安装。


另外，对应版本的[Pytorch](https://pytorch.org/)包（版本>=2.0）也要安装好。推荐使用 `pip` 安装python依赖包。请一定认真安装对应的cuda和torch版本，本测试环境中使用的cuda版本为12.1。pytorch详细安装教程请参考[官方网站](https://pytorch.org/)。

```bash
pip install 'torch<2.4' --index-url https://download.pytorch.org/whl/cu121
```

如果只需要进行模型推理（预测任务），也可以只安装CPU版本的Pytorch。

```bash
pip install 'torch<2.4' --index-url https://download.pytorch.org/whl/cpu
```

下一步，克隆本仓库，并安装其他模型训练/推理所需的依赖。

```bash
git clone --recursive https://github.com/zhangtaolab/Plant_DNA_LLMs
cd Plant_DNA_LLMs
python3 -m pip install -r requirements.txt
```

（可选步骤）如果需要训练[Mamba](https://github.com/state-spaces/mamba)模型，还需要安装以下这些额外的依赖；此外，DNA Mamba模型的训练和推理必须使用NVIDIA显卡。  
注意Mamba模型最低支持Nvidia 10系以上的显卡（如GTX1660s，RTX2080, RTX3060, RTX4070等，不支持GTX1080Ti等20系列以下的显卡）

```bash
pip install 'causal-conv1d<=1.3'
pip install 'mamba-ssm<2'
```

#### 1.4 安装git-lfs
`glt-lfs`主要用于下载大语言模型和数据集，`git-lfs`安装教程参考 [git-lfs install](https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing)。如果安装成功
```bash
$ git lfs version
```
会显示类似的结果
```bash
git-lfs/3.3.0 (GitHub; linux amd64; go 1.19.8)
```