
import torch
import gradio as gr

# 定义任务和模型
model_dir = "zhangtaolab"
BASE_MODEL = ['plant-dnabert','plant-nucleotide-transformer','plant-dnagpt',
               'plant-dnagemma','dnabert2','nucleotide-transformer-v2-100m','agront-1b']
tokenizers = ['BPE','singlebase','6mer']

TASK_MAP = {
    "promoter": {
        "title": "Core promoter prediction by plant DNAMamba",
        "desc": "<p>Predict whether the input sequence is a active core promoter.</p>",
        "id2label": {0: 'Not promoter', 1: 'Core promoter'}
    },
    "conservation": {
        "title": "Sequence conservation prediction by plant DNAMamba",
        "desc": "<p>Predict whether the input sequence is conserved sequence.</p>",
        "id2label": {0: 'Not conserved', 1: 'Conserved'}
    },
    "H3K27ac": {
        "title": "Histone modification (H3K27ac) prediction by plant DNAMamba",
        "desc": "<p>Predict whether the input sequence is from H3K27ac histone modification regions (Data from ChIP-hub).</p>",
        "id2label": {0: 'Not H3K27ac', 1: 'H3K27ac'}
    },
    "H3K27me3": {
        "title": "Histone modification (H3K27me3) prediction by plant DNAMamba",
        "desc": "<p>Predict whether the input sequence is from H3K27me3 histone modification regions (Data from ChIP-hub).</p>",
        "id2label": {0: 'Not H3K27me3', 1: 'H3K27me3'}
    },
    "H3K4me3": {
        "title": "Histone modification (H3K4me3) prediction by plant DNAMamba",
        "desc": "<p>Predict whether the input sequence is from H3K4me3 histone modification regions (Data from ChIP-hub).</p>",
        "id2label": {0: 'Not H3K4me3', 1: 'H3K4me3'}
    },
    "lncRNAs": {
        "title": "Putative IncRNAs prediction by plant DNAMamba",
        "desc": "<p>Predict whether the input sequence is an IncRNA (Data from GreeNC).</p>",
        "id2label": {0: 'Not IncRNA', 1: 'IncRNA'}
    },
    "open_chromatin": {
        "title": "Open chromation regions prediction by plant DNAMamba",
        "desc": "<p>Predict whether the input sequence is from open chromatin regions (detected by DNase-seq, ATAC-seq, MH-seq, etc.).</p>",
        "id2label": {0: 'Not open chromatin', 1: 'Full open chromatin', 2: 'Partial open chromatin'}
    },
    "promoter_strength_leaf": {
        "title": "Promoters strength prediction",
        "desc": "<p>Predict the (core) promoter strength in tobacco leaves based on STARR-seq data.</p>",
        "id2label": {0: 'Promoter strength in tobacco leaves'}
    },
    "promoter_strength_protoplast": {
        "title": "Promoters strength prediction",
        "desc": "<p>Predict the (core) promoter strength in maize protoplasts based on STARR-seq data.</p>",
        "id2label": {0: 'Promoter strength in maize protoplasts'}
    },
}

REGRESSION_TASKS = ["promoter_strength_leaf", "promoter_strength_protoplast"]

####################################################################################################
def update_task(task):
    desc = TASK_MAP[task]["desc"]
    information = gr.HTML("<H3>Task Information</H3>" + "<p>" + desc + "</p>")
    return information

def inference(seq,task,model,tokenizer_type,source):
    if not seq:
        gr.Warning("No sequence provided, use the default sequence.")
        seq = placeholder
    #选择模型来源
    if source=='huggingface':
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
    else:
        from modelscope import AutoModelForSequenceClassification,AutoTokenizer    
    #选择模型和分词方式
    if model.startswith('plant'):
        model_name = f'zhangtaolab/{model}-{tokenizer_type}-{task}'
    else:
        model_name = f'zhangtaolab/{model}-{task}'
    model = AutoModelForSequenceClassification.from_pretrained(model_name,ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Inference
    inputs = tokenizer(seq, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits

    if task in REGRESSION_TASKS:
        result = logits.item()
        return None, result
    else:
        probs = torch.nn.functional.softmax(logits, dim=-1)
        labels = TASK_MAP[task]["id2label"].values()
        result = {label: prob.item() for label, prob in zip(labels, probs[0])}
        return result, None


placeholder="TATAGAGAATCACATATCGAGCTCGAGAGAGAGAGAGAGAGAGAGAGAGACGGAGAAACGAAAACCTGAAAGCCAACCGGCATATGAACCGTCGCGCAGG"
css = """
footer{display:none !important}
textarea::placeholder{opacity: 0.25}
"""
demo = gr.Blocks(title="Prediction of plant DNA downstream tasks by plant DNAMamba model",
                 css=css).queue(default_concurrency_limit=3)

with demo:
    gr.HTML(
        """
        <h1 style="text-align: center;">Prediction of plant DNA downstream tasks by plant DNAMamba model</h1>
        """
    )
    with gr.Row():
        drop1 = gr.Dropdown(choices=TASK_MAP.keys(),
                            label="Select Task",
                            interactive=True,
                            value="promoter")
        drop2 = gr.Dropdown(choices=BASE_MODEL,
                            label="Select Model",
                            interactive=True,
                            value=BASE_MODEL[0])
    with gr.Row():
        drop3 = gr.Dropdown(choices=['BPE','singlebase','6mer'],
                    label="Select Tokenizer",
                    interactive=True,
                    value="BPE")
   
        drop4 = gr.Dropdown(choices=['huggingface','modelscope'],
                            label="Select Source",
                            interactive=True,
                            value="modelscope")

    # 读取模型
    task = drop1.value
    model_name = drop2.value
    tokenizer = drop3.value
    source = drop4.value

    desc = TASK_MAP[task]["desc"]
    information = gr.HTML("<H3>Task Information</H3>" + "<p>" + desc + "</p>")
   
    with gr.Column():
        input = gr.TextArea(label="Input Sequence",
                            placeholder="TATAGAGAATCACATATCGAGCTCGAGAGAGAGAGAGAGAGAGAGAGAGACGGAGAAACGAAAACCTGAAAGCCAACCGGCATATGAACCGTCGCGCAGG",
                            lines=4,
                            interactive=True,
                            show_copy_button=True)
    with gr.Row():
        submit_btn = gr.Button("Predict", variant='primary')
        clear_btn = gr.ClearButton(components=[drop1, drop2, input])
   
    output1 = gr.Label(label="Classification")
    output2 = gr.Number(label="Predicted Score (Regression)")
    submit_btn.click(inference, inputs=[input,drop1,drop2,drop3,drop4], outputs=[output1, output2])
    # 监听下拉菜单变化
    drop1.change(update_task, inputs=[drop1], outputs=[information])

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')
