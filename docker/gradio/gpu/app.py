import gradio as gr
from pdllib import ModelInference, task_map
from pdllib.models import base_model, tokenizer_type, data_source


####################################################################################################

# 定义任务和模型
model_dir = "zhangtaolab"
TASK_MAP = task_map
BASE_MODEL = base_model
TOKENIZER = tokenizer_type
SOURCE = data_source

####################################################################################################

# 输入序列检查

def check_sequence(sequence):
    if set(sequence.upper()) - set('ACGTN') != set():
        return False, "Unsupported characters found in the sequence."

    length = len(sequence)
    if length < 20:
        if length == 0:
            return True, "No sequence provided, use the default sequence."
        else:
            return False, "Sequence length is " + str(length) + " bp, which is lower than the minimum length (20)."
    elif length > 6000:
        return False, "Sequence length is " + str(length) + " bp, which is greater than the maximum length (6000)."

    return True, ""

def get_sequences(input):
    texts = str(input).split("\n")
    input_type = "plain"
    sequence = ""
    for i, text in enumerate(texts):
        if i == 0:
            if text.startswith(">"):
                input_type = "fasta"
                continue
        if input_type == "plain":
            sequence = "".join(texts)
            status, reason = check_sequence(sequence)
            if status:
                if reason:
                    sequence = placeholder
                    gr.Warning(reason)
                return sequence
            else:
                raise gr.Error(reason)
                return ""
        else:
            if text.startswith(">"):
                status, reason = check_sequence(sequence)
                if status:
                    if reason:
                        sequence = placeholder
                        gr.Warning(reason)
                    return sequence
                else:
                    raise gr.Error(reason)
                    return ""
            else:
                sequence += text
    if sequence:
        return sequence

####################################################################################################

# 模型推理函数

def update_task(task):
    desc = TASK_MAP[task]["desc"]
    information = gr.HTML("<H3>Task Information</H3>" + "<p>" + desc + "</p>")
    return information

def run_inference(text, task, model, tokenizer_type, source, max_length=512):
    # 检查序列是否符合要求
    sequence = get_sequences(text)
    if not sequence:
        return {}

    # 读取模型名称
    if model.startswith('plant'):
        model_path = f'zhangtaolab/{model}-{tokenizer_type}-{task}'
    else:
        model_path = f'zhangtaolab/{model}-{task}'

    # 初始化ModelInference对象
    model = ModelInference(
        model_path=model_path,
        source=source,
        device="auto",
        max_token=max_length
    )

    # 进行预测
    results = model.predict([sequence])

    # 导出结果
    for result in results:
        if TASK_MAP[task]["datatype"].endswith("regression"):
            return None, result['probability'].values()[0]
        else:
            return result['probability'], None

####################################################################################################

placeholder="TATAGAGAATCACATATCGAGCTCGAGAGAGAGAGAGAGAGAGAGAGAGACGGAGAAACGAAAACCTGAAAGCCAACCGGCATATGAACCGTCGCGCAGG"
css = """
textarea::placeholder{opacity: 0.25}
"""
demo = gr.Blocks(title="Prediction of plant DNA downstream tasks by plant DNA LLMs",
                 css=css).queue(default_concurrency_limit=5)

# Gradio GUI for selecting task, model, and tokenizer
with demo:
    gr.HTML(
        """
        <h1 style="text-align: center;">Prediction of plant DNA downstream tasks by plant DNA LLMs</h1>
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
        drop3 = gr.Dropdown(choices=TOKENIZER,
                    label="Select Tokenizer",
                    interactive=True,
                    value="BPE")

        drop4 = gr.Dropdown(choices=SOURCE,
                            label="Select Source",
                            interactive=True,
                            value="modelscope")

    # 读取模型
    task = drop1.value

    # 读取任务信息
    desc = TASK_MAP[task]["desc"]
    information = gr.HTML("<H3>Task Information</H3>" + "<p>" + desc + "</p>")

    # Gradio GUI for input and output
    with gr.Column():
        input = gr.TextArea(label="Input Sequence",
                            placeholder="TATAGAGAATCACATATCGAGCTCGAGAGAGAGAGAGAGAGAGAGAGAGACGGAGAAACGAAAACCTGAAAGCCAACCGGCATATGAACCGTCGCGCAGG",
                            lines=4,
                            interactive=True,
                            show_copy_button=True)
    with gr.Row():
        submit_btn = gr.Button("Predict", variant='primary')
        clear_btn = gr.ClearButton(components=[drop1, drop2, input])

    output1 = gr.Label(label="Predicted Label (Classification)")
    output2 = gr.Number(label="Predicted Score (Regression)")
    submit_btn.click(run_inference,
                     inputs=[input, drop1, drop2, drop3, drop4],
                     outputs=[output1, output2])

    # 监听下拉菜单变化
    drop1.change(update_task,
                 inputs=[drop1],
                 outputs=[information])


if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')
