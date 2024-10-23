import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import marimo as mo
from pdllib.api import plant_llms_inference

model_path = "zhangtaolab/plant-dnamamba-BPE-promoter"
sequence = "GGGAAAAAGTGAACTCCATTGTTTTTTCACGCTAAGCAGACCACAATTGCTGCTTGGTACGAAAAGAAAACCGAACCCTTTCACCCACGCACAACTCCATCTCCATTAGCATGGACAGAACACCGTAGATTGAACGCGGGAGGCAACAGGCTAAATCGTCCGTTCAGCCAAAACGGAATCATGGGCTGTTTTTCCAGAAGGCTCCGTGTCGTGTGGTTGTGGTCCAAAAACGAAAAAGAAAGAAAAAAGAAAACCCTTCCCAAGACGTGAAGAAAAGCAATGCGATGCTGATGCACGTTA"

res = plant_llms_inference(model_path, sequence, source='modelscope', device='auto', max_length=512)
print(res)
res = plant_llms_inference(model_path, sequence, source='huggingface', device='auto', max_length=512)
print(res)