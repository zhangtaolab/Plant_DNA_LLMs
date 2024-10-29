import marimo

__generated_with = "0.9.12"
app = marimo.App(width="medium")


@app.cell
def __(__file__):
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import marimo as mo
    from pdllib.api import plant_llms_inference
    import pandas as pd
    return mo, os, pd, plant_llms_inference, sys


@app.cell
def __(pd):
    model_df = pd.read_excel("./plant_DNA_LLMs_finetune_list.xlsx")
    return (model_df,)


@app.cell
def __():
    # model_df
    return


@app.cell
def __(model_df):
    tasks = model_df.Task.unique()
    return (tasks,)


@app.cell
def __(mo, tasks):
    task_dropdown = mo.ui.dropdown(tasks, value='open chromatin', label='Predict Task:')
    return (task_dropdown,)


@app.cell
def __():
    # task_dropdown
    return


@app.cell
def __(model_df, task_dropdown):
    models = model_df[model_df.Task == task_dropdown.value ].Model.unique()
    return (models,)


@app.cell
def __(mo, models):
    model_dropdown = mo.ui.dropdown(models, label='Model:', value='Plant DNABERT')
    return (model_dropdown,)


@app.cell
def __():
    # model_dropdown
    return


@app.cell
def __(model_df, model_dropdown, task_dropdown):
    tokenizers = model_df[ (model_df.Task == task_dropdown.value) & (model_df.Model == model_dropdown.value) ].Tokenzier.unique()
    return (tokenizers,)


@app.cell
def __(mo, tokenizers):
    tokenizer_dropdown = mo.ui.dropdown(tokenizers, label='Tokenizer', value='BPE')
    return (tokenizer_dropdown,)


@app.cell
def __(mo):
    source_dropdown = mo.ui.dropdown({'modelscope':'modelscope',
                                        'huggingface':'huggingface'
                                     },label='Model Source:', value='modelscope')
    return (source_dropdown,)


@app.cell
def __(mo):
    dnaseq_entry_box = mo.ui.text_area(placeholder='GGGCAGCGGTTACACCTTAATCGACACGACTCTCGGCAACGGATATCTCGGCTCTTGCATCGATGAAGAACGTAGCAAAATGCGATACCTGGTGTGAATTGCAGAATCCCGCGAACCATCGAGTTTTTGAACGCAAGTTGCGCCCGAAGCCTTCTGACGGAGGGCACGTCTGCCTGGGCGTCACGCCAAAAGACACTCCCAACACCCCCCCGCGGGGCGAGGGACGTGGCGTCTGGCCCCCCGCGCTGCAGGGCGAGGTGGGCCGAAGCAGGGGCTGCCGGCGAACCGCGTCGGACGCAACACGTGGTGGGCGACATCAAGTTGTTCTCGGTGCAGCGTCCCGGCGCGCGGCCGGCCATTCGGCCCTAAGGACCCATCGAGCGACCGAGCTTGCCCTCGGACCACGACCCCAGGTCAGTCGGGACTACCCGCTGAGTTTAAGCATATAAATAAGCGGAGGAGAAGAAACTTACGAGGATTCCCCTAGTAACGGCGAGCGAACCGGGAGCAGCCCAGCTTGAGAATCGGGCGGCCTCGCCGCCCGAATTGTAGTCTGGAGAGGCGT', full_width=True, label='DNA Sequence:', rows=5)
    return (dnaseq_entry_box,)


@app.cell
def __(
    dnaseq_entry_box,
    mo,
    model_dropdown,
    source_dropdown,
    task_dropdown,
    tokenizer_dropdown,
):
    hstack=mo.hstack([task_dropdown, model_dropdown, tokenizer_dropdown,source_dropdown],align='center', justify='center')
    mo.vstack([dnaseq_entry_box, hstack])
    return (hstack,)


@app.cell
def __(model_df, model_dropdown, task_dropdown, tokenizer_dropdown):
    model_name = model_df[ (model_df.Task == task_dropdown.value) & (model_df.Model == model_dropdown.value) & (model_df.Tokenzier==tokenizer_dropdown.value)].Name
    return (model_name,)


@app.cell
def __():
    # print(task_dropdown.value, model_dropdown.value, tokenizer_dropdown.value, source_dropdown.value, model_name.values[0])
    return


@app.cell
def __(dnaseq_entry_box):
    dnaseq = ''
    if dnaseq_entry_box.value:
        dnaseq = dnaseq_entry_box.value
    else:
        dnaseq='GGGCAGCGGTTACACCTTAATCGACACGACTCTCGGCAACGGATATCTCG\
        GCTCTTGCATCGATGAAGAACGTAGCAAAATGCGATACCTGGTGTGAATTGCAGAAT\
        CCCGCGAACCATCGAGTTTTTGAACGCAAGTTGCGCCCGAAGCCTTCTGACGGA\
        GGGCACGTCTGCCTGGGCGTCACGCCAAAAGACACTCCCAACACCCCCCCGCGGGGC\
        GAGGGACGTGGCGTCTGGCCCCCCGCGCTGCAGGGCGAGGTGGGCCGAAGCAGGGGCTGCC\
        GGCGAACCGCGTCGGACGCAACACGTGGTGGGCGACATCAAGTTGTTCTCGGTGCAGCGT\
        CCCGGCGCGCGGCCGGCCATTCGGCCCTAAGGACCCATCGAGCGACCGAGCTTGCCCTCG\
        GACCACGACCCCAGGTCAGTCGGGACTACCCGCTGAGTTTAAGCATATAAATAAGCGGAGGAG\
        AAGAAACTTACGAGGATTCCCCTAGTAACGGCGAGCGAACCGGGAGCAGCCCAGCTTGA\
        GAATCGGGCGGCCTCGCCGCCCGAATTGTAGTCTGGAGAGGCGT'
        print("use default sequence")
    return (dnaseq,)


@app.cell
def __(dnaseq, model_name, plant_llms_inference, source_dropdown):
    res = plant_llms_inference(model_name.values[0], dnaseq, source=source_dropdown.value, device='auto', max_length=512)
    return (res,)


@app.cell
def __(res):
    res
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
