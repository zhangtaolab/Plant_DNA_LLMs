<h1>
  <p align="center">
  PDLLMs: A group of tailored DNA large language models for analyzing plant genomes
  </p>
</h1>

<p align="center">
  <b>English | </b>
  <a href="README_zh.md" target="_blank"><b>简体中文</b></a>
</p>

## 0. Demo for plant DNA LLMs prediction

![demo](imgs/plantllm.gif)


Online prediction of other models and prediction tasks can be found [here](https://finetune.plantllm.org/).

## 1. Environment

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
pip install -r requirements.txt
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

## 2. Pretrain from scratch

To pretrain our DNA models from scratch, please first download the desired pretrained models from [HuggingFace](https://huggingface.co/zhangtaolab) or [ModelScope](https://www.modelscope.cn/organization/zhangtaolab) to local. You can use `git clone` (which may require `git-lfs` to be installed) to retrieve the model or directly download the model from the website.

In the activated `llms` python environment, use the `model_pretrain_from_scratch.py` script to pretrain a model use your own dataset. 

Before training the model, please prepare a training dataset that contains the splited sequences (length shorted than 2000 bp) from the target genomes. (Detailed information can be found in the supplemental notes out our [manuscript](https://www.cell.com/molecular-plant/fulltext/S1674-2052(24)00390-3))

The dataset file should look like [this](https://huggingface.co/datasets/zhangtaolab/plant-reference-genomes).

* Here is the [pretrain models list](docs/en/resources/pretrain_models.md)

We use DNAGPT model as an example to perform the pretrain.

```bash
# prepare a output directory
mkdir pretrain
# download pretrain model
git clone https://huggingface.co/zhangtaolab/plant-dnagpt-BPE models/plant-dnagpt-BPE
# prepare your own dataset for pretraining, data can be stored at the data directory
# example: data/pretrain_data.txt
```

* Note: If downloading from huggingface encounters network error, please try to download model from ModelScope or change to the accelerate mirror before downloading.
```bash
# Download with git
git clone https://hf-mirror.com/[organization_name/repo_name]
# Download with huggingface-cli
export HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download [organization_name/repo_name]
```

After preparing the model and dataset, using the following script to pretrain the model.

```bash
python model_pretrain_from_scratch.py \
    --model_name_or_path models/plant-dnagpt-BPE \
    --train_data data/pretrain_data.txt \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 24 \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --bf16 \
    --logging_strategy steps \
    --logging_steps 100 \
    --save_strategy steps \
    --save_steps 500 \
    --output_dir pretrain/dnagpt-BPE_updated
```

In this script:  
1. `--model_name_or_path`: Path to the foundation model you downloaded
2. `--train_data`: Path to the train dataset
3. `--per_device_train_batch_size`: Batch size for training model
4. `--gradient_accumulation_steps`: Number of updates steps to accumulate the gradients for, before performing a backward/update pass
5. `--num_train_epochs`: Epoch for training model (also you can train model with steps, then you should change the strategies for save, logging and evaluation)
6. `--learning_rate`: Learning rate for training model
7. `--warmup_ratio`: Ratio of total training steps used for a linear warmup from 0 to `learning_rate`
8. `--bf16`: Use bf16 precision for training
9. `--logging_strategy`: Strategy for logging training information, can be `epoch` or `steps`
10. `--logging_steps`: Steps for logging training information
11. `--save_strategy`: Strategy for saving model, can be `epoch` or `steps`
12. `--save_steps`: Steps for saving model checkpoints
13. `--output_dir`: Where to save the pretrained model

Detailed descriptions of the arguments can be referred [here](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments).

Finally, wait for the progress bar completed, and the pretrained model will be saved in the `pretrain/dnagpt-BPE_updated` directory. In this directory, there will be checkpoint directories, a runs directory, and a saved pretrained model.

## 3. Fine-tune

To fine-tune the plant DNA LLMs, please first download the desired models from [HuggingFace](https://huggingface.co/zhangtaolab) or [ModelScope](https://www.modelscope.cn/organization/zhangtaolab) to local. You can use `git clone` (which may require `git-lfs` to be installed) to retrieve the model or directly download the model from the website.

In the activated `llms` python environment, use the `model_finetune.py` script to fine-tune a model for downstream task.  

Our script accepts `.csv` format data (separated by `,`) as input, when preparing the training data, please make sure the data contain a header and at least these two columns:
```csv
sequence,label
```
Where `sequence` is the input sequence, and `label` is the corresponding label for the sequence.

We also provide several plant genomic datasets for fine-tuning on the [HuggingFace](https://huggingface.co/zhangtaolab) and [ModelScope](https://www.modelscope.cn/organization/zhangtaolab).

* Here is the [pretrain models list](docs/en/resources/pretrain_models.md)

We use Plant DNAGPT model as example to fine-tune a model for active core promoter prediction.

First download a pretrain model and corresponding dataset from HuggingFace or ModelScope:

```bash
# prepare a output directory
mkdir finetune
# download pretrain model
git clone https://huggingface.co/zhangtaolab/plant-dnagpt-BPE models/plant-dnagpt-BPE
# download train dataset
git clone https://huggingface.co/datasets/zhangtaolab/plant-multi-species-core-promoters data/plant-multi-species-core-promoters
```

* Note: If downloading from huggingface encounters network error, please try to download model/dataset from ModelScope or change to the accelerate mirror before downloading.
```bash
# Download with git
git clone https://hf-mirror.com/[organization_name/repo_name]
# Download with huggingface-cli
export HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download [organization_name/repo_name]
```

After preparing the model and dataset, using the following script to finetune model (here is a promoter prediction example)

```bash
python model_finetune.py \
    --model_name_or_path models/plant-dnagpt-BPE \
    --train_data data/plant-multi-species-core-promoters/train.csv \
    --test_data data/plant-multi-species-core-promoters/test.csv \
    --eval_data data/plant-multi-species-core-promoters/dev.csv \
    --train_task classification \
    --labels 'Not promoter;Core promoter' \
    --run_name plant_dnagpt_BPE_promoter \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --load_best_model_at_end \
    --metric_for_best_model 'f1' \
    --save_strategy epoch \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --output_dir finetune/plant-dnagpt-BPE-promoter
```

In this script:  
1. `--model_name_or_path`: Path to the foundation model you downloaded
2. `--train_data`: Path to the train dataset
3. `--test_data`: Path to the test dataset, omit it if no test data available
4. `--dev_data`: Path to the validation dataset, omit it if no validation data available
5. `--train_task`: Determine the task type, should be classification, multi-classification or regression
6. `--labels`: Set the labels for classification task, separated by `;`
7. `--run_name`: Name of the fine-tuned model
8. `--per_device_train_batch_size`: Batch size for training model
9. `--per_device_eval_batch_size`: Batch size for evaluating model
10. `--learning_rate`: Learning rate for training model
11. `--num_train_epochs`: Epoch for training model (also you can train model with steps, then you should change the strategies for save, logging and evaluation)
12. `--load_best_model_at_end`: Whether to load the model with the best performance on the evaluated data, default is `True`
13. `--metric_for_best_model`: Use which metric to determine the best model, default is `loss`, can be `accuracy`, `precison`, `recall`, `f1` or `matthews_correlation` for classification task, and `r2` or `spearmanr` for regression task
14. `--save_strategy`: Strategy for saving model, can be `epoch` or `steps`
15. `--logging_strategy`: Strategy for logging training information, can be `epoch` or `steps`
16. `--evaluation_strategy`: Strategy for evaluating model, can be `epoch` or `steps`
17. `--output_dir`: Where to save the fine-tuned model

Detailed descriptions of the arguments can be referred [here](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments).

Finally, wait for the progress bar completed, and the fine-tuned model will be saved in the `plant-dnagpt-BPE-promoter` directory. In this directory, there will be a checkpoint directory, a runs directory, and a saved fine-tuning model.

## 4. Inference

To use a fine-tuned model for inference, please first download the desired models from [HuggingFace](https://huggingface.co/zhangtaolab) or [ModelScope](https://www.modelscope.cn/organization/zhangtaolab) to local or provide a model trained by yourself.

* Here is the [finetune models list](docs/en/resources/finetune_models.md)

We use Plant DNAGPT model as example to predict active core promoter in plants.

First download a fine-tuned model and corresponding dataset from HuggingFace or ModelScope

```bash
# prepare a work directory
mkdir inference
# download fine-tuned model
git clone https://huggingface.co/zhangtaolab/plant-dnagpt-BPE-promoter models/plant-dnagpt-BPE-promoter
# download train dataset
git clone https://huggingface.co/datasets/zhangtaolab/plant-multi-species-core-promoters data/plant-multi-species-core-promoters
```

We provide a script named `model_inference.py` for model inference.  
Here is an example that use the script to predict histone modification:

```bash
# (method 1) Inference with local model, directly input a sequence
python model_inference.py -m models/plant-dnagpt-BPE-promoter -s 'TTACTAAATTTATAACGATTTTTTATCTAACTTTAGCTCATCAATCTTTACCGTGTCAAAATTTAGTGCCAAGAAGCAGACATGGCCCGATGATCTTTTACCCTGTTTTCATAGCTCGCGAGCCGCGACCTGTGTCCAACCTCAACGGTCACTGCAGTCCCAGCACCTCAGCAGCCTGCGCCTGCCATACCCCCTCCCCCACCCACCCACACACACCATCCGGGCCCACGGTGGGACCCAGATGTCATGCGCTGTACGGGCGAGCAACTAGCCCCCACCTCTTCCCAAGAGGCAAAACCT'

# (method 2) Inference with local model, provide a file contains multiple sequences to predict
python model_inference.py -m models/plant-dnagpt-BPE-promoter -f data/plant-multi-species-core-promoters/test.csv -o inference/promoter_predict_results.txt

# (method 3) Inference with an online model (Auto download the model trained by us from huggingface or modelscope)
python model_inference.py -m zhangtaolab/plant-dnagpt-BPE-promoter -ms huggingface -s 'GGGAAAAAGTGAACTCCATTGTTTTTTCACGCTAAGCAGACCACAATTGCTGCTTGGTACGAAAAGAAAACCGAACCCTTTCACCCACGCACAACTCCATCTCCATTAGCATGGACAGAACACCGTAGATTGAACGCGGGAGGCAACAGGCTAAATCGTCCGTTCAGCCAAAACGGAATCATGGGCTGTTTTTCCAGAAGGCTCCGTGTCGTGTGGTTGTGGTCCAAAAACGAAAAAGAAAGAAAAAAGAAAACCCTTCCCAAGACGTGAAGAAAAGCAATGCGATGCTGATGCACGTTA'
```

In this script:
1. `-m`: Path to the fine-tuned model that is used for inference
2. `-s`: Input DNA sequence, only nucleotide A, C, G, T, N are acceptable
3. `-f`: Input file that contain multiple sequences, one line for each sequence. If you want to keep more information, file with `,` of `\t` separator is acceptable, but a header contains `sequence` column must be specified.
4. `-ms`: Download the model from `huggingface` or `modelscope` if the model is not local. The format of model name is `zhangtaolab/model-name`, users can copy model name here:
![copy](imgs/huggingface_copy.png)

Output results contains the original sequence, input sequence length. If the task type is classification, predicted label and probability of each label will provide; If the task type is regression, a predicted score will provide.


### 5. Docker implementation for model inference

Environment deployment for LLMs may be an arduous job. To simplify this process, we also provide a docker version of our model inference code.

The images of the docker version are [here](https://hub.docker.com/r/zhangtaolab/plant_llms_inference), and the usage of docker implementation is shown below.

#### Inference using GPU

For GPU inference (with Nvidia GPU), please pull the image with `gpu` tag, and make sure your computer has install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

First download a finetune model from Huggingface or ModelScope, here we use Plant DNAMamba model as an example to predict active core promoters。

```bash
# prepare a work directory
mkdir LLM_inference
cd LLM_inference
git clone https://huggingface.co/zhangtaolab/plant-dnamamba-BPE-promoter
```

Then download the corresponding dataset, and if users have their own data, users can also prepare a custom dataset based on the previously mentioned inference data format.

```bash
git clone https://huggingface.co/datasets/zhangtaolab/plant-multi-species-core-promoters
```

Once the model and dataset are ready, pull our model inference image from docker and test if it works.

```bash
docker pull zhangtaolab/plant_llms_inference:gpu
docker run --runtime=nvidia --gpus=all -v ./:/home/llms zhangtaolab/plant_llms_inference:gpu -h
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

If the preceding information is displayed, the image is downloaded and the inference script can run normally. 
Inference is performed below using previously prepared models and datasets.

```bash
docker run --runtime=nvidia --gpus=all -v ./:/home/llms zhangtaolab/plant_llms_inference:gpu -m /home/llms/plant-dnamamba-BPE-promoter -f /home/llms/plant-multi-species-core-promoters/test.csv -o /home/llms/predict_results.txt
```

After the inference progress bar is completed, see the output file `predict_results.txt` in the current local directory, which saves the prediction results corresponding to each sequence in the input file.

#### Inference using CPU

For CPU inference,  please pull the image with `cpu` tag, this image support computer without NVIDIA GPU, such as cpu-only or Apple M-series Silicon. (Note that Inference of DNAMamba model is not supported in CPU mode)

First download a finetune model from Huggingface or ModelScope, here we use Plant DNAGPT model as an example to predict active core promoters。

```bash
# prepare a work directory
mkdir LLM_inference
cd LLM_inference
git clone https://huggingface.co/zhangtaolab/plant-dnagpt-BPE-promoter
```

Then download the corresponding dataset, and if users have their own data, users can also prepare a custom dataset based on the previously mentioned inference data format.

```bash
git clone https://huggingface.co/datasets/zhangtaolab/plant-multi-species-core-promoters
```

Once the model and dataset are ready, pull our model inference image from docker and test if it works.

```bash
docker pull zhangtaolab/plant_llms_inference:cpu
docker run -v ./:/home/llms zhangtaolab/plant_llms_inference:cpu -h
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

If the preceding information is displayed, the image is downloaded and the inference script can run normally. 
Inference is performed below using previously prepared models and datasets.

```bash
docker run -v ./:/home/llms zhangtaolab/plant_llms_inference:cpu -m /home/llms/plant-dnagpt-BPE-promoter -f /home/llms/plant-multi-species-core-promoters/test.csv -o /home/llms/predict_results.txt
```

After the inference progress bar is completed, see the output file `predict_results.txt` in the current local directory, which saves the prediction results corresponding to each sequence in the input file.

* The detailed usage is the same as the section [Inference](#3-inference).

### Online prediction platform

In order to facilitate users to use the model to predict DNA analysis tasks, we also provide online prediction platforms.

Please refer to [online prediction platform](docs/en/resources/platforms.md)


### 6. Inference API for developers

For developers who want to use our inference code in the Jupyter Notebook or other places, we developed a simple API package in the `pdllib`, which allows users directly call the inference function.

Besides, we provide a Demo that shows the usage of our API, see [notebook/inference_demo.ipynb](notebook/inference_demo.ipynb).


### Citation

* Liu GQ, Chen L, Wu YC, Han YS, Bao Y, Zhang T\*. [PDLLMs: A group of tailored DNA large language models for analyzing plant genomes](https://doi.org/10.1016/j.molp.2024.12.006). ***Molecular Plant*** 2025, 18(2):175-178
