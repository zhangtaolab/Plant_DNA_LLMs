import sys
from os import path, makedirs
import json
import random
import time
import argparse
import numpy as np
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, pipeline


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
                                            "  docker run -v /local:/container zhangtaolab/plant_llms_inference:cpu -m model_path -f seqfile.csv -o output.txt\n"
                                            "  docker run -v /local:/container zhangtaolab/plant_llms_inference:cpu -m model_path -s 'ATCGGATCTCGACAGT' -o output.txt\n")

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
    if path.exists(args.model):
        args.source = "local"
        from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
    else:
        if args.source == "huggingface":
            from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
        elif args.source == "modelscope":
            from modelscope import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
        else:
            from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

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

    if "dnamamba" in model_name:
        print("Mamba model is not supported to inference on a non-Nvidia GPU currently.")
        sys.exit(1)
        # if num_labels > 1:
        #     model = MambaSequenceClassification.from_pretrained(args.model, num_classes=num_labels)
        # else:
        #     model = MambaSequenceRegression.from_pretrained(args.model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=num_labels,
            trust_remote_code=True
        )
    model.config.num_labels = num_labels
    model.config.id2label = id2label

    if "dnabert" in model_name:
        max_length = 512
    elif "agront" in model_name:
        max_length = 1024
    elif "dnagemma" in model_name:
        max_length = 1024
    elif "dnagpt" in model_name:
        max_length = 1024
    elif "dnamamba" in model_name:
        max_length = 2048
    elif "plant_nt" in model_name:
        max_length = 2048
    elif "nt_v2_100m" in model_name:
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
