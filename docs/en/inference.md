To use a fine-tuned model for inference, please first download the desired models from [HuggingFace](https://huggingface.co/zhangtaolab) or [ModelScope](https://www.modelscope.cn/organization/zhangtaolab) to local or provide a model trained by yourself.

* Here is the [finetune models list](resources/finetune_models.md)

We use Plant DNAGPT model as example to predict active core promoter in plants.

First download a fine-tuned model and corresponding dataset from HuggingFace or ModelScope

```bash
# prepare a work directory
mkdir LLM_inference
cd LLM_inference
# download fine-tuned model
git clone https://huggingface.co/zhangtaolab/plant-dnagpt-BPE-promoter
# download train dataset
git clone https://huggingface.co/zhangtaolab/plant-multi-species-core-promoters
```

We provide a script named `model_inference.py` for model inference.  
Here is an example that use the script to predict histone modification:

```bash
# (method 1) Inference with local model, directly input a sequence
python model_inference.py -m ./plant-dnagpt-BPE-promoter -s 'TTACTAAATTTATAACGATTTTTTATCTAACTTTAGCTCATCAATCTTTACCGTGTCAAAATTTAGTGCCAAGAAGCAGACATGGCCCGATGATCTTTTACCCTGTTTTCATAGCTCGCGAGCCGCGACCTGTGTCCAACCTCAACGGTCACTGCAGTCCCAGCACCTCAGCAGCCTGCGCCTGCCATACCCCCTCCCCCACCCACCCACACACACCATCCGGGCCCACGGTGGGACCCAGATGTCATGCGCTGTACGGGCGAGCAACTAGCCCCCACCTCTTCCCAAGAGGCAAAACCT'

# (method 2) Inference with local model, provide a file contains multiple sequences to predict
python model_inference.py -m ./plant-dnagpt-BPE-promoter -f ./plant-multi-species-core-promoters/test.csv -o promoter_predict_results.txt

# (method 3) Inference with an online model (Auto download the model trained by us from huggingface or modelscope)
python model_inference.py -m zhangtaolab/plant-dnagpt-BPE-promoter -ms huggingface -s 'GGGAAAAAGTGAACTCCATTGTTTTTTCACGCTAAGCAGACCACAATTGCTGCTTGGTACGAAAAGAAAACCGAACCCTTTCACCCACGCACAACTCCATCTCCATTAGCATGGACAGAACACCGTAGATTGAACGCGGGAGGCAACAGGCTAAATCGTCCGTTCAGCCAAAACGGAATCATGGGCTGTTTTTCCAGAAGGCTCCGTGTCGTGTGGTTGTGGTCCAAAAACGAAAAAGAAAGAAAAAAGAAAACCCTTCCCAAGACGTGAAGAAAAGCAATGCGATGCTGATGCACGTTA'
```

In this script:
1. `-m`: Path to the fine-tuned model that is used for inference
2. `-s`: Input DNA sequence, only nucleotide A, C, G, T, N are acceptable
3. `-f`: Input file that contain multiple sequences, one line for each sequence. If you want to keep more information, file with `,` of `\t` separator is acceptable, but a header contains `sequence` column must be specified.
4. `-ms`: Download the model from `huggingface` or `modelscope` if the model is not local. The format of model name is `zhangtaolab/model-name`, users can copy model name here:
![copy](resources/huggingface_copy.png)

Output results contains the original sequence, input sequence length. If the task type is classification, predicted label and probability of each label will provide; If the task type is regression, a predicted score will provide.

