[Anaconda](https://docs.anaconda.com/free/anaconda/install/) package manager is recommended for building the training environment. For pre-train and fine-tune models, please ensure that you have a Nvidia GPU and the corresponding drivers are installed. For inference, devices without Nvidia GPU (CPU only, AMD GPU, Apple Silion, etc.) are also acceptable.

#### 1.1 Download and install [Anaconda](https://www.anaconda.com/download) package manager

#### 1.2 Create environment (We trained the models with python 3.11)

```bash
conda create -n llms python=3.11
conda activate llms
```

#### 1.3 Install dependencies

If you want to pre-train or fine-tune models, make sure you are using Nvidia GPU(s).  
Install [Nvidia driver](https://www.nvidia.com/download/index.aspx) and corresponding version of [CUDA driver](https://developer.nvidia.com/cuda-downloads) (> 11.0, we used CUDA 12.1).  

Also [Pytorch](https://pytorch.org/) (>=2.0) with corresponding CUDA version should also install.  
We recommend to use `pip` to install python packages that needed. Please be sure to install the corresponding CUDA and Torch versions carefully, the CUDA version used in this test environment is 12.1. Please refer to [Official Website](https://pytorch.org/) for the detailed installation tutorial of pytorch.

```bash
pip install 'torch<2.4' --index-url https://download.pytorch.org/whl/cu121
```

If you just want to use models for inference (prediction), you can install Pytorch GPU version (above) or install Pytorch CPU version if your machine has no Nvidia GPU.

```bash
pip install 'torch<2.4' --index-url https://download.pytorch.org/whl/cpu
```

Next install other required dependencies.
```bash
git clone --recursive https://github.com/zhangtaolab/Plant_DNA_LLMs
cd Plant_DNA_LLMs
python3 -m pip install -r requirements.txt
```

(Optional) If you want to train a [mamba](https://github.com/state-spaces/mamba) model, you need to install several extra dependencies, also you should have a Nvidia GPU.

```bash
pip install 'causal-conv1d<=1.3'
pip install 'mamba-ssm<2'
```

#### 1.4 Install git-lfs
`glt-lfs` is required for download large models and datasets，`git-lfs` installation can be refer to [git-lfs install](https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing).

If `git-lfs` is installed, run the following command
```bash
$ git lfs version
```
will get message like this
```bash
git-lfs/3.3.0 (GitHub; linux amd64; go 1.19.8)
```