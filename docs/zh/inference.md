在使用微调模型推理（预测任务）前，请先下载已经我们提供的微调模型（[ModelScope](https://www.modelscope.cn/organization/zhangtaolab) 或 [HuggingFace](https://huggingface.co/zhangtaolab)）到本地，或使用前面提供的脚本自己训练一个微调模型用于推理。

* 微调模型的列表可参考 [微调模型列表](resources/finetune_models.md)

我们同样提供了模型推理的脚本 `model_inference.py` ，下面是一个预测植物活性启动子的例子：

```bash
# （方法1）直接输入序列
python model_inference.py -m plant-dnagpt-BPE-promoter -s 'TTACTAAATTTATAACGATTTTTTATCTAACTTTAGCTCATCAATCTTTACCGTGTCAAAATTTAGTGCCAAGAAGCAGACATGGCCCGATGATCTTTTACCCTGTTTTCATAGCTCGCGAGCCGCGACCTGTGTCCAACCTCAACGGTCACTGCAGTCCCAGCACCTCAGCAGCCTGCGCCTGCCATACCCCCTCCCCCACCCACCCACACACACCATCCGGGCCCACGGTGGGACCCAGATGTCATGCGCTGTACGGGCGAGCAACTAGCCCCCACCTCTTCCCAAGAGGCAAAACCT'

# （方法2）提供一个包含多条待预测序列的文件用于推理
python model_inference.py -m plant-dnagpt-BPE-promoter -f plant-multi-species-core-promoters/test.csv -o promoter_predict_results.txt
```

以上的命令中，不同参数的介绍如下：  
1. `-m`: 微调模型的路径
2. `-s`: 待预测的序列, 只支持包含A, C, G, T, N碱基的序列
3. `-f`: 包含多条待预测序列的文件，一行对应一条序列。如果需要保留更多的信息，使用 `,` 或者 `\t` 分隔符，但是包含表头的 `sequence` 列必须指定

输出结果会包含原始序列，序列的长度，如果是分类任务，会返回预测的分类结果及其对应的预测可能性；如果是回归任务，会返回预测的得分。