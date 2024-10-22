前面提到的大语言模型训练所需环境的构建比较繁琐，需要安装大量的依赖。为了精简这一过程，我们提供了一个基于docker的镜像，方便用户快速实现模型推理。
 
docker镜像可以在[这里查看](https://hub.docker.com/r/zhangtaolab/plant_llms_inference)，此外，使用docker进行模型推理的例子已在下方展示。

#### 基于显卡（GPU）进行推理

我们提供了2种模型推理的镜像，对于有NVIDIA显卡的设备，可以拉取 `gpu` 标签的docker镜像，同时确保电脑里已经正确安装了 [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

首先先从ModelScope或Huggingface下载微调模型，这里以Plant DNAMamba模型为例，预测活性启动子。

```bash
# 准备一个工作目录
mkdir LLM_inference
cd LLM_inference
git clone https://modelscope.cn/models/zhangtaolab/plant-dnamamba-BPE-promoter
```

接着下载对应的数据集，如果用户有自己的数据，也可以根据前面提到的推理数据格式准备自定义数据集。

```bash
git clone https://modelscope.cn/datasets/zhangtaolab/plant-multi-species-core-promoters
```

模型和数据集准备完毕后，从docker拉取我们的模型推理镜像，并测试是否可以正常运行。

```bash
# 如果下载失败，可能是网络问题，请重新运行命令多试几次
docker pull cr.bioinfor.eu.org/zhangtaolab/plant_llms_inference:gpu
# 下载完成后，运行以下命令检查镜像能否运行成功
docker run --runtime=nvidia --gpus=all -v ./:/home/llms cr.bioinfor.eu.org/zhangtaolab/plant_llms_inference:gpu -h
```

```bash
usage: inference.py [-h] [-v] -m MODEL [-f FILE] [-s SEQUENCE] [-t THRESHOLD]
                    [-l MAX_LENGTH] [-bs BATCH_SIZE] [-p SAMPLE] [-seed SEED]
                    [-d {cpu,gpu,mps,auto}] [-o OUTFILE] [-n]

Script for Plant DNA Large Language Models (LLMs) inference

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -m MODEL              Model path (should contain both model and tokenizer)
  -f FILE               File contains sequences that need to be classified
  -s SEQUENCE           One sequence that need to be classified
  -t THRESHOLD          Threshold for defining as True class (Default: 0.5)
  -l MAX_LENGTH         Max length of tokenized sequence (Default: 512)
  -bs BATCH_SIZE        Batch size for classification (Default: 1)
  -p SAMPLE             Subsampling for testing (Default: 1e7)
  -seed SEED            Random seed for subsampling (Default: None)
  -d {cpu,gpu,mps,auto}
                        Choose CPU or GPU to do inference (require specific
                        drivers) (Default: auto)
  -o OUTFILE            Prediction results (Default: stdout)
  -n                    Whether or not save the runtime locally (Default:
                        False)

Example:
  docker run --runtime=nvidia --gpus=all -v /local:/container zhangtaolab/plant_llms_inference:gpu -m model_path -f seqfile.csv -o output.txt
  docker run --runtime=nvidia --gpus=all -v /local:/container zhangtaolab/plant_llms_inference:gpu -m model_path -s 'ATCGGATCTCGACAGT' -o output.txt
```

成功出现以上信息，说明镜像下载完整，且可以正常运行推理脚本。  
下面使用先前准备好的模型和数据集进行推理。

```bash
docker run --runtime=nvidia --gpus=all -v ./:/home/llms cr.bioinfor.eu.org/zhangtaolab/plant_llms_inference:gpu -m /home/llms/plant-dnamamba-BPE-promoter -f /home/llms/plant-multi-species-core-promoters/test.csv -o /home/llms/predict_results.txt
```

稍等推理进度条完成后，在本地当前目录浏览`predict_results.txt`，该文件保存了输入文件中每条序列对应的预测结果。

#### 基于CPU进行推理

对于没有显卡的机器，可以拉取 `cpu` 标签的docker镜像，该镜像基本适用于所有设备，包括纯CPU、苹果芯片设备等，不过需要注意的是，该镜像暂时无法推理DNA Mamba模型。

首先先下载从ModelScope或Huggingface下载微调模型，这里以Plant DNAGPT模型为例，预测活性启动子。

```bash
# 准备一个工作目录
mkdir LLM_inference
cd LLM_inference
git clone https://modelscope.cn/models/zhangtaolab/plant-dnagpt-BPE-promoter
```

接着下载对应的数据集，如果用户有自己的数据，也可以根据前面提到的推理数据格式准备自定义数据集。

```bash
git clone https://modelscope.cn/datasets/zhangtaolab/plant-multi-species-core-promoters
```

模型和数据集准备完毕后，从docker拉取我们的模型推理镜像，并测试是否可以正常运行。

```bash
docker pull cr.bioinfor.eu.org/zhangtaolab/plant_llms_inference:cpu
docker run -v ./:/home/llms cr.bioinfor.eu.org/zhangtaolab/plant_llms_inference:cpu -h
```

```bash
usage: inference.py [-h] [-v] -m MODEL [-f FILE] [-s SEQUENCE] [-t THRESHOLD]
                    [-l MAX_LENGTH] [-bs BATCH_SIZE] [-p SAMPLE] [-seed SEED]
                    [-d {cpu,gpu,mps,auto}] [-o OUTFILE] [-n]

Script for Plant DNA Large Language Models (LLMs) inference

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -m MODEL              Model path (should contain both model and tokenizer)
  -f FILE               File contains sequences that need to be classified
  -s SEQUENCE           One sequence that need to be classified
  -t THRESHOLD          Threshold for defining as True class (Default: 0.5)
  -l MAX_LENGTH         Max length of tokenized sequence (Default: 512)
  -bs BATCH_SIZE        Batch size for classification (Default: 1)
  -p SAMPLE             Subsampling for testing (Default: 1e7)
  -seed SEED            Random seed for subsampling (Default: None)
  -d {cpu,gpu,mps,auto}
                        Choose CPU or GPU to do inference (require specific
                        drivers) (Default: auto)
  -o OUTFILE            Prediction results (Default: stdout)
  -n                    Whether or not save the runtime locally (Default:
                        False)

Example:
  docker run -v /local:/container zhangtaolab/plant_llms_inference:gpu -m model_path -f seqfile.csv -o output.txt
  docker run -v /local:/container zhangtaolab/plant_llms_inference:gpu -m model_path -s 'ATCGGATCTCGACAGT' -o output.txt
```

成功出现以上信息，说明镜像下载完整，且可以正常运行推理脚本。  
下面使用先前准备好的模型和数据集进行推理。

```bash
docker run -v ./:/home/llms cr.bioinfor.eu.org/zhangtaolab/plant_llms_inference:cpu -m /home/llms/plant-dnagpt-BPE-promoter -f /home/llms/plant-multi-species-core-promoters/test.csv -o /home/llms/predict_results.txt
```

稍等推理进度条完成后，在本地当前目录浏览`predict_results.txt`，该文件保存了输入文件中每条序列对应的预测结果。

* 脚本中其他参数的说明可以参考 [模型推理](inference.md) 中的使用说明。

### 在线预测平台

为了方便用户使用模型预测DNA分析任务，我们也提供了在线的预测平台。

请参考：[在线预测列表](resources/platforms.md)