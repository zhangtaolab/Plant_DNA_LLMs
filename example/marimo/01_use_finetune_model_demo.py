import marimo

__generated_with = "0.9.12"
app = marimo.App(width="medium")


@app.cell
def __(__file__):
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    import marimo as mo
    from pdllib.api import plant_llms_inference
    return mo, os, plant_llms_inference, sys


@app.cell
def __():
    model_path = "zhangtaolab/plant-dnabert-BPE-promoter"
    sequence = "GGGAAAAAGTGAACTCCATTGTTTTTTCACGCTAAGCAGACCACAATTGCTGCTTGGTACGAAAAGAAAACCGAACCCTTTCACCCACGCACAACTCCATCTCCATTAGCATGGACAGAACACCGTAGATTGAACGCGGGAGGCAACAGGCTAAATCGTCCGTTCAGCCAAAACGGAATCATGGGCTGTTTTTCCAGAAGGCTCCGTGTCGTGTGGTTGTGGTCCAAAAACGAAAAAGAAAGAAAAAAGAAAACCCTTCCCAAGACGTGAAGAAAAGCAATGCGATGCTGATGCACGTTA"
    return model_path, sequence


@app.cell
def __(model_path, plant_llms_inference, sequence):
    res = plant_llms_inference(model_path, sequence, source='modelscope', device='auto', max_length=512)
    return (res,)


@app.cell
def __(res):
    print(res)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
