Environment deployment for LLMs may be an arduous job. To simplify this process, we also provide a docker version of our model inference code.

The images of the docker version are [here](https://hub.docker.com/r/zhangtaolab/plant_llms_inference), and the usage of docker implementation is shown below.

#### Inference using GPU

For GPU inference (with Nvidia GPU), please pull the image with `gpu` tag, and make sure your computer has install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

First download a finetune model from Huggingface or ModelScope, here we use Plant DNAMamba model as an example to predict active core promoters。

```bash
# prepare a work directory
mkdir LLM_inference
cd LLM_inference
git clone https://modelscope.cn/models/zhangtaolab/plant-dnamamba-BPE-promoter
```

Then download the corresponding dataset, and if users have their own data, users can also prepare a custom dataset based on the previously mentioned inference data format.

```bash
git clone https://modelscope.cn/datasets/zhangtaolab/plant-multi-species-core-promoters
```

Once the model and dataset are ready, pull our model inference image from docker and test if it works.

```bash
docker pull zhangtaolab/plant_llms_inference:gpu
docker run --runtime=nvidia --gpus=all -v /Local_path:/Path_in_container zhangtaolab/plant_llms_inference:gpu -h
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
git clone https://modelscope.cn/models/zhangtaolab/plant-dnagpt-BPE-promoter
```

Then download the corresponding dataset, and if users have their own data, users can also prepare a custom dataset based on the previously mentioned inference data format.

```bash
git clone https://modelscope.cn/datasets/zhangtaolab/plant-multi-species-core-promoters
```

Once the model and dataset are ready, pull our model inference image from docker and test if it works.

```bash
docker pull zhangtaolab/plant_llms_inference:cpu
docker run -v /Local_path:/Path_in_container zhangtaolab/plant_llms_inference:cpu -h
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

* The detailed usage is the same as the section [Inference](inference.md).

### Online prediction platform

In order to facilitate users to use the model to predict DNA analysis tasks, we also provide online prediction platforms.

Please refer to [online prediction platform](resources/platforms.md)