import sys
from os import path, makedirs
import json
import random
import time
import argparse
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, pipeline
from transformers.utils import ModelOutput

from collections import namedtuple
from dataclasses import dataclass, field, asdict

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
    mamba_available = True
except:
    mamba_available = False

task_map = {
    "promoters": {
        "labels": ["Not promoter", "Core promoter"]
    },
    "H3K27ac": {
        "labels": ["Not H3K27ac", "H3K27ac"]
    },
    "H3K27me3": {
        "labels": ["Not H3K27me3", "H3K27me3"]
    },
    "H3K4me3": {
        "labels": ["Not H3K4me3", "H3K4me3"]
    },
    "DNA_methylation": {
        "labels": ["Not methylated", "Methylated"]
    },
    "conservation": {
        "labels": ["Not conserved", "Conserved"]
    },
    "lncRNAs": {
        "labels": ["Not lncRNA", "lncRNA"]
    },
    "open_chromatin": {
        "labels": ["Not open chromatin", "Full open chromatin", "Partial open chromatin"]
    },
    "pro_str_leaf": {
        "labels": ["Promoter strength in tobacco leaves"]
    },
    "pro_str_protoplast": {
        "labels": ["Promoter strength in maize protoplasts"]
    }
}


if mamba_available:
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


def check_sequence(sequence):
    if len(sequence) < 20 or len(sequence) > 6000:
        return False
    elif set(sequence.upper()) - set('ACGTN') != set():
        return False
    else:
        return True

def load_seqfile(file, batch_size=1, sample=10000000, seed=None):
    seqs = []
    batch = []
    filtered = 0

    if file.endswith(".fa") or file.endswith(".fasta"):
        with open(file) as fi:
            for line in fi:
                if line.startswith(">"):
                    continue
                seq = line.strip().upper()
                if check_sequence(seq):
                    batch.append(seq)
                else:
                    filtered += 1
                if len(batch) == batch_size:
                    seqs.append(batch)
                    batch = []
    else:
        if file.endswith(".csv"):
            sep = ","
        else:
            sep = "\t"
        cnt = 0
        seq_idx = -1
        with open(file) as fi:
            for line in fi:
                info = line.strip().split(sep)
                if cnt == 0:
                    if "sequence" in info:
                        seq_idx = info.index("sequence")
                        continue
                if seq_idx == -1:
                    seq = info[-1].upper()
                    if check_sequence(seq):
                        batch.append(seq)
                else:
                    seq = info[seq_idx].upper()
                    if check_sequence(seq):
                        batch.append(seq)
                    else:
                        filtered += 1
                if len(batch) == batch_size:
                    seqs.append(batch)
                    batch = []
    if filtered > 0:
        print("Filtered %s sequence(s) due to unsupported chars or length." % filtered)

    len_seqs = sum([len(batch) for batch in seqs])
    if len_seqs > sample:
        seqs = [seq for batch in seqs for seq in batch]
        if seed is not None:
            random.seed(seed)
        random.shuffle(seqs)
        seqs = [seqs[i:min(i+batch_size, sample)] for i in range(0, sample, batch_size)]

    return seqs


def get_options():

    parser = argparse.ArgumentParser(description="Script for Plant DNA Large Language Models (LLMs) inference", 
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="Example:\n"
                                            "  python model_inference.py -m model_path -f seqfile.csv -o output.txt\n"
                                            "  python model_inference.py -m model_path -s 'ATCGGATCTCGACAGT' -o output.txt\n")

    parser.add_argument('-v', '--version', action='version', version='%(prog)s '+'v0.1')

    parser.add_argument('-m', dest='model', required=True, type=str, help='Model path (should contain both model and tokenizer)')

    parser.add_argument('-ms', dest='source', type=str, choices=['huggingface', 'modelscope', 'local'], default='huggingface', help='Download source of the model')

    parser.add_argument('-f', dest='file', default=None, type=str, help='File contains sequences that need to be classified')

    parser.add_argument('-s', dest='sequence', type=str, default=None, help='One sequence that need to be classified')

    parser.add_argument('-t', dest='threshold', type=float, default=0.5, help='Threshold for defining as True class (Default: 0.5)')

    parser.add_argument('-l', dest='max_length', type=int, default=512, help='Max length of tokenized sequence (Default: 512)')

    parser.add_argument('-bs', dest='batch_size', type=int, default=1, help='Batch size for classification (Default: 1)')

    parser.add_argument('-p', dest='sample', type=int, default=10000000, help='Subsampling for testing (Default: 1e7)')

    parser.add_argument('-seed', dest='seed', type=int, default=None, help='Random seed for subsampling (Default: None)')

    parser.add_argument('-d', dest='device', type=str, choices=['cpu','gpu','mps','auto'], default='auto', help='Choose CPU or GPU to do inference (require specific drivers) (Default: auto)')

    parser.add_argument('-o', dest='outfile', type=str, default=None, help='Prediction results (Default: stdout)')

    parser.add_argument('-n', dest='time', action='store_true', default=False, help='Whether or not save the runtime locally (Default: False)')

    return parser

def check_options(parser):

    args = parser.parse_args()

    if args.file is not None and args.sequence is not None:
        print("Please input either sequence or file, not both.")
        print("Overwrite sequence with file input.")
        args.sequence = None

    if args.sequence is None:
        if args.file is None:
            print("Please input sequence or file that need to be classified.")
            parser.print_help()
            sys.exit(1)
        else:
            if not path.exists(path.abspath(path.expanduser(args.file))):
                print("Can not located sequence file, please input full path of sequence file.")
                parser.print_help()
                sys.exit(1)

    if args.file is None:
        if args.sequence is None:
            print("Please input sequence or file that need to be classified.")
            parser.print_help()
            sys.exit(1)
        else:
            if len(args.sequence) < 20 or len(args.sequence) > 6000:
                print("Sequence length should be between 20 and 6000.")
                parser.print_help()
                sys.exit(1)
            elif set(args.sequence.upper()) - set('ACGTN') != set():
                print("Sequence should only contain A, C, G, T or N.")
                parser.print_help()
                sys.exit(1)
    elif not path.exists(path.abspath(path.expanduser(args.file))):
        print("Can not located sequence file, please input full path of sequence file.")
        parser.print_help()
        sys.exit(1)

    if args.outfile is not None:
        if not path.exists(path.dirname(path.abspath(path.expanduser(args.outfile)))):
            print('Make directory: %s\n' % path.dirname(path.abspath(path.expanduser(args.outfile))))
            makedirs(path.dirname(path.abspath(path.expanduser(args.outfile))))

    return args


def main():

    args = check_options(get_options())

    threshold = args.threshold
    batch_size = args.batch_size
    subsample = args.sample
    seed = args.seed
    max_token = args.max_length
    save_time = args.time

    if args.file:
        seqfile = path.abspath(path.expanduser(args.file))
        seqs = load_seqfile(seqfile, batch_size=batch_size, sample=subsample, seed=seed)
    else:
        seqfile = None
        if args.sequence:
            sequence = args.sequence
            seqs = [[sequence]]
        else:
            sequence = None
            print("Please input sequence or file that need to be classified.")
            sys.exit(1)

    if args.device == 'auto':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device('mps')
    elif args.device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif args.device == 'mps':
        if hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    print("Use device:", device)

    if args.outfile is not None:
        outf = open(args.outfile, 'w')

    # define model source
    if os.path.exists(args.model):
        args.source = "local"
        prefix = ""
    else:
        if args.source == "huggingface":
            prefix = "https://huggingface.co/"
        elif args.source == "modelscope":
            prefix = "https://modelscope.cn/models/"
        else:
            prefix = "https://huggingface.co/"
    args.model = prefix + args.model

    config = AutoConfig.from_pretrained(args.model, from_pretrained=True)
    model_name = config._name_or_path

    print("Model:", model_name)
    id2label = config.id2label
    num_labels = len(id2label)
    if id2label[0] == "LABEL_0":
        for task in task_map:
            if task in model_name:
                num_labels = len(task_map[task]['labels'])
                id2label = {i: task_map[task]['labels'][i] for i in range(num_labels)}
        if id2label[0] == "LABEL_0":
            if num_labels == 2:
                    id2label = {0: 'False', 1: 'True'}
            elif num_labels == 3:
                id2label = {0: 'None', 1: 'Full', 2: 'Partial'}
            else:
                id2label = {i: str(i) for i in range(num_labels)}

    if "dnamamba" in model_name.lower():
        if mamba_available:
            if num_labels > 1:
                model = MambaSequenceClassification.from_pretrained(args.model, num_classes=num_labels)
            else:
                model = MambaSequenceRegression.from_pretrained(args.model)
            model.device = device
        else:
            sys.exit("Mamba model is not installed or your device is not supported.")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=num_labels,
            trust_remote_code=True
        )
    model.config.num_labels = num_labels
    model.config.id2label = id2label

    if "dnabert" in model_name.lower():
        max_length = 512
    elif "agront" in model_name.lower():
        max_length = 1024
    elif "dnagemma" in model_name.lower():
        max_length = 1024
    elif "dnagpt" in model_name.lower():
        max_length = 1024
    elif "dnamamba" in model_name.lower():
        max_length = 2048
    elif "plant_nt" in model_name.lower():
        max_length = 2048
    elif "nt_v2_100m" in model_name.lower():
        max_length = 2048
    else:
        max_length = 512

    max_length = min(max_length, max(128, max_token))
    if max_length != max_token:
        print("New max token length:", max_length)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    len_seqs = sum([len(batch) for batch in seqs])
    if num_labels > 1:
        pipe = pipeline('text-classification', model=model, tokenizer=tokenizer,
                        trust_remote_code=True, device=device, top_k=None)
        if args.outfile is not None:
            print("Sequence", "Length", "Label", "Probability", sep="\t", file=outf)
            progress_bar = tqdm(total=len(seqs))
        start = time.time()
        for batch in seqs:
            results = pipe(batch, truncation=True, padding='max_length', max_length=max_length)
            for i, result in enumerate(results):
                l = ''
                label_res = [x['label'] for x in result]
                score_res = [x['score'] for x in result]
                best = score_res.index(max(score_res))
                if num_labels == 2:
                    score = score_res[best]
                    if score > threshold:
                        l = label_res[best]
                    else:
                        l = label_res[1-best]
                else:
                    l = label_res[best]
                proba = {label_res[j]: score_res[j] for j in range(len(label_res))}
                proba = {k: proba[k] for k in id2label.values()}
                if args.outfile is None:
                    print("Sequence:", batch[i], "Seq_length:", len(batch[i]), "Label:", l, "Probability:", proba, sep="\n")
                else:
                    print(batch[i], len(batch[i]), l, proba, sep="\t", file=outf)
            if args.outfile is not None:
                progress_bar.update(1)
    else:
        pipe = pipeline('text-classification', model=model, tokenizer=tokenizer,
                        trust_remote_code=True, device=device, function_to_apply="none")
        if args.outfile is not None:
            print("Sequence", "Length", "Score", sep="\t", file=outf)
            progress_bar = tqdm(total=len(seqs))
        start = time.time()
        for batch in seqs:
            results = pipe(batch, truncation=True, padding='max_length', max_length=max_length)
            for i, result in enumerate(results):
                if args.outfile is None:
                    print("Sequence:", batch[i], "Seq_length:", len(batch[i]), "Score:", result['score'], sep="\n")
                else:
                    print(batch[i], len(batch[i]), result['score'], sep="\t", file=outf)
            if args.outfile is not None:
                progress_bar.update(1)

    end = time.time()
    diff = end - start
    items_per_second = len_seqs / diff

    print(f'time: {diff:.3f} s | {items_per_second:.3f} it/s')
    if save_time and args.outfile:
        outfile = path.abspath(path.dirname(args.outfile)) + "/runtime_" + str(end) + ".txt"
        with open(outfile, "w") as ot:
            print(f'time: {diff:.3f} s | {items_per_second:.3f} it/s', file=ot)

    if args.outfile is not None:
        outf.close()


if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        sys.stderr.write("User interrupt\n")
        sys.exit(0)
