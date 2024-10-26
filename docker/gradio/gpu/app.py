import torch
import gradio as gr
from dataclasses import dataclass, field, asdict
from torch import nn
import json
from transformers.utils import ModelOutput
import numpy as np
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

# 定义任务和模型
model_dir = "zhangtaolab"
BASE_MODEL = ['plant-dnamamba','plant-dnabert','plant-nucleotide-transformer','plant-dnagpt',
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
#mamba
@dataclass
class MambaConfig:
    d_model: int = 768
    n_layer: int = 24
    vocab_size: int = 8000
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True

    def __init__(self,
                _name_or_path="Plant_DNAMamba",
                architectures=["MambaForCausalLM"],
                bos_token_id=0,
                conv_kernel=4,
                d_inner=1536,
                d_model=768,
                eos_token_id=0,
                expand=2,
                fused_add_norm=True,
                hidden_act="silu",
                hidden_size=768,
                initializer_range=0.02,
                intermediate_size=1536,
                layer_norm_epsilon=1e-05,
                model_type="mamba",
                n_layer=24,
                numb_hidden_layers=24,
                pad_token_id=0,
                pad_vocab_size_multiple=8,
                problem_type="single_label_classification",
                rescale_prenorm_residual=False,
                residual_in_fp32=True,
                rms_norm=True,
                state_size=16,
                task_specific_params={"text-generation": {"do_sample": True, "max_length": 50}},
                tie_embeddings=True,
                time_step_floor=0.0001,
                time_step_init_scheme="random",
                time_step_max=0.1,
                time_step_min=0.001,
                time_step_rank=48,
                time_step_scale=1.0,
                torch_dtype="float32",
                transformers_version="4.39.1",
                use_bias=True,
                use_cache=True,
                use_conv_bias=True,
                ssm_cfg={},
                vocab_size=8000,
                **kwargs,
                ):
        self._name_or_path = _name_or_path
        self.architectures = architectures
        self.bos_token_id = bos_token_id
        self.conv_kernel = conv_kernel
        self.d_inner = d_inner
        self.d_model = d_model
        self.eos_token_id = eos_token_id
        self.expand = expand
        self.fused_add_norm = fused_add_norm
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.model_type = model_type
        self.n_layer = n_layer
        self.numb_hidden_layers = numb_hidden_layers
        self.pad_token_id = pad_token_id
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.problem_type = problem_type
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.rms_norm = rms_norm
        self.state_size = state_size
        self.task_specific_params = task_specific_params
        self.tie_embeddings = tie_embeddings
        self.time_step_floor = time_step_floor
        self.time_step_init_scheme = time_step_init_scheme
        self.time_step_max = time_step_max
        self.time_step_min = time_step_min
        self.time_step_rank = time_step_rank
        self.time_step_scale = time_step_scale
        self.torch_dtype = torch_dtype
        self.transformers_version = transformers_version
        self.use_bias = use_bias
        self.use_cache = use_cache
        self.use_conv_bias = use_conv_bias
        self.ssm_cfg = ssm_cfg
        self.vocab_size = vocab_size
        self._commit_hash = None

    def to_json_string(self):
        return json.dumps(asdict(self))

    def to_dict(self):
        return asdict(self)


class MambaClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes, **kwargs):
        super(MambaClassificationHead, self).__init__()
        self.classification_head = nn.Linear(d_model, num_classes, **kwargs)

    def forward(self, hidden_states):
        return self.classification_head(hidden_states)


class MambaSequenceClassification(MambaLMHeadModel):
    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
        num_classes=2,
    ) -> None:
        super().__init__(config, initializer_cfg, device, dtype)

        self.classification_head = MambaClassificationHead(d_model=config.d_model, num_classes=num_classes)

        del self.lm_head

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.backbone(input_ids)

        mean_hidden_states = hidden_states.mean(dim=1)

        logits = self.classification_head(mean_hidden_states)

        if labels is None:
            # ClassificationOutput = namedtuple("ClassificationOutput", ["logits"])
            return ModelOutput(logits=logits)
        # else:
        #     ClassificationOutput = namedtuple("ClassificationOutput", ["loss", "logits"])

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        # return ClassificationOutput(loss=loss, logits=logits)
        return ModelOutput(loss=loss, logits=logits)

    def can_generate(self):
        return False

    def predict(self, text, tokenizer, id2label=None):
        input_ids = torch.tensor(tokenizer(text)['input_ids'], device='cuda')[None]
        with torch.no_grad():
            logits = self.forward(input_ids).logits[0]
            label = np.argmax(logits.cpu().numpy())

        if id2label is not None:
            return id2label[label]
        else:
            return label

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, num_classes=2, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, num_classes=num_classes, **kwargs)

        model_state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        model.load_state_dict(model_state_dict, strict=False)

        # print("Newly initialized embedding:", set(model.state_dict().keys()) - set(model_state_dict.keys()))
        return model


class MambaSequenceRegression(MambaLMHeadModel):
    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(config, initializer_cfg, device, dtype)

        self.classification_head = MambaClassificationHead(d_model=config.d_model, num_classes=1)

        del self.lm_head

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.backbone(input_ids)

        mean_hidden_states = hidden_states.mean(dim=1)

        logits = self.classification_head(mean_hidden_states)

        if labels is None:
            # ClassificationOutput = namedtuple("ClassificationOutput", ["logits"])
            return ModelOutput(logits=logits)
        # else:
        #     ClassificationOutput = namedtuple("ClassificationOutput", ["loss", "logits"])

        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())

        # return ClassificationOutput(loss=loss, logits=logits)
        return ModelOutput(loss=loss, logits=logits)

    def can_generate(self):
        return False

    def predict(self, text, tokenizer, id2label=None):
        input_ids = torch.tensor(tokenizer(text)['input_ids'], device='cuda')[None]
        with torch.no_grad():
            logits = self.forward(input_ids).logits[0]
            label = logits.cpu().numpy()

        if id2label is not None:
            return id2label[label]
        else:
            return label

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)

        model = cls(config, device=device, dtype=dtype, **kwargs)

        model_state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        model.load_state_dict(model_state_dict, strict=False)

        # print("Newly initialized embedding:", set(model.state_dict().keys()) - set(model_state_dict.keys()))
        return model

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
        from transformers import AutoModelForSequenceClassification,AutoTokenizer,AutoConfig,AutoModel
    else:
        from modelscope import AutoModelForSequenceClassification,AutoTokenizer,AutoConfig,AutoModel
    #当选择mamba模型时使用自定义模型
    if "mamba" in model:
        model_name = f'zhangtaolab/{model}-{tokenizer_type}-{task}'
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if task in REGRESSION_TASKS:
            model = MambaSequenceRegression.from_pretrained(config._name_or_path).to('cuda')
        else:
            model = MambaSequenceClassification.from_pretrained(config._name_or_path,num_classes=len(TASK_MAP[task]['id2label'])).to('cuda')
    else:  
    #选择模型和分词方式
        if model.startswith('plant'):
            model_name = f'zhangtaolab/{model}-{tokenizer_type}-{task}'
        else:
            model_name = f'zhangtaolab/{model}-{task}'
        model = AutoModelForSequenceClassification.from_pretrained(model_name,ignore_mismatched_sizes=True).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Inference
    inputs = tokenizer(seq, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: value.to('cuda') for key, value in inputs.items()}  # 移动输入到 GPU
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
