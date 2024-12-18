# coding=utf-8
#! /usr/bin/env python

import os
# Uncomment this if you want to use specific directory for cache
# os.environ['HF_HOME'] = '/mnt/data'
# Uncomment these when P2P and InfiniBand are not available during training with multiple GPUs
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_IB_DISABLE'] = '1'
# Uncomment this to use the accelerate mirror for Hugging Face downloads
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, HfArgumentParser
from datasets import Dataset, DatasetDict, load_dataset


############################################################
## Arguments ###############################################

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./models/plantdnabert")
    tokenizer_path: Optional[str] = field(default=None)
    train_task: Optional[str] = field(default='classification')
    load_checkpoint: Optional[str] = field(default=None)
    # select model source
    source: Optional[str] = field(default="huggingface")

@dataclass
class DataArguments:
    train_data: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data: Optional[str] = field(default=None, metadata={"help": "Path to the valid data."})
    test_data: Optional[str] = field(default=None, metadata={"help": "Path to the test data."})
    split: float = field(default=0.1, metadata={"help": "Test split"})
    samples: Optional[int] = field(default=1e8)
    key: str = field(default='sequence', metadata={"help": "Feature name"})
    kmer: int = field(default=-1, metadata={"help": "k-mer for DNABERT model"})
    labels: str = field(default='No;Yes', metadata={"help": "Labels"})
    shuffle: bool = field(default=False)

@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="runs")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)
    logging_steps: Optional[int] = field(default=500)
    logging_strategy: str = field(default='epoch'),
    save_steps: Optional[int] = field(default=200)
    save_strategy: str = field(default='epoch'),
    eval_steps: Optional[int] = field(default=None)
    eval_strategy: str = field(default='epoch'),
    # eval_accumulation_steps: Optional[int] = field(default=None),
    # warmup_steps: int = field(default=50)
    warmup_ratio: float = field(default=0.05)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-5)
    save_total_limit: int = field(default=5)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    seed: int = field(default=7)


############################################################
## Functions ###############################################

# Load data (fasta or csv format)
def load_fasta_data(data, labels=None, data_type='', sample=1e8, seed=7):
    # check if the data is existed
    if not data:
        return {}
    if not os.path.exists(data):
        return {}

    # Generate data dictionary
    dic = {'idx': [], 'sequence': [], 'label': []}
    idx = 0
    with open(data) as infile:
        for line in tqdm(infile, desc=data_type):
            if line.startswith(">"):
                name = line.strip()[1:].split("|")[0]
                name = "_".join(name.split("_")[:-1])
                label_info = line.strip()[1:].split("|")[1:]
                label = label_info[0]
            else:
                seq = line.strip()
                dic['idx'].append(name)
                dic['sequence'].append(seq)
                if labels:
                    dic['label'].append(int(label))
                else:
                    dic['label'].append(float(label))
                idx += 1
    # random sampling for fasta format data
    if sample < idx:
        random.seed(seed)
        print('Downsampling %s data to %s.' % (data_type, sample))
        indices = random.sample(range(idx), k=sample)
        sampled_dic = {'idx': [], 'sequence': [], 'label': []}
        # for label in labels:
        #     sampled_dic.update({label: []})
        for i in tqdm(indices):
            sampled_dic['idx'].append(dic['idx'][i])
            sampled_dic['sequence'].append(dic['sequence'][i])
            sampled_dic['label'].append(dic['label'][i])
        return sampled_dic
    else:
        return dic


def seq2kmer(seqs, k):
    all_kmers = []
    for seq in seqs:
        kmer = [seq[x:x+k].upper() for x in range(len(seq)+1-k)]
        kmers = " ".join(kmer)
        all_kmers.append(kmers)
    return all_kmers


def evaluate_metrics(task):
    task = task.lower()
    if task == "regression":
        from pdllib.metrics import regression_metrics
        compute_metrics = regression_metrics()

    elif task == "multi-classification":
        from pdllib.metrics import multi_classification_metrics
        compute_metrics = multi_classification_metrics()

    elif task.startswith("multi-label"):
        from pdllib.metrics import multi_labels_metrics
        compute_metrics = multi_labels_metrics()
    else:
        from pdllib.metrics import classification_metrics
        compute_metrics = classification_metrics()

    return compute_metrics


############################################################
## Trainer ###############################################

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # define model source
    if os.path.exists(model_args.model_name_or_path):
        model_args.source = "local"
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
    else:
        if model_args.source == "huggingface":
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
        elif model_args.source == "modelscope":
            from modelscope import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
        else:
            print("Unknown Source type, using Hugging Face transformers library.")
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

    # load tokenizer
    if not model_args.tokenizer_path:
        model_args.tokenizer_path = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        use_fast=True,
        trust_remote_code=True
    )

    # define datasets and data collator
    labels = data_args.labels.split(";")
    # load csv format data
    if data_args.train_data.endswith(".csv"):
        if data_args.eval_data and data_args.test_data:
            data_files = {'train': data_args.train_data, 'dev': data_args.eval_data, 'test': data_args.test_data}
            dataset = load_dataset('csv', data_files=data_files)
        elif data_args.eval_data:
            data_files = {'train': data_args.train_data, 'dev': data_args.eval_data}
            dataset = load_dataset('csv', data_files=data_files)
        elif data_args.test_data:
            data_files = {'train': data_args.train_data, 'test': data_args.test_data}
            dataset = load_dataset('csv', data_files=data_files)
        else:
            dataset = load_dataset('csv', data_files=data_args.train_data)
            dataset = dataset['train'].train_test_split(test_size=data_args.split)
        if data_args.shuffle:
            dataset["train"] = dataset["train"].shuffle(training_args.seed)
    # load fasta format data (from AgroNT plant-genomic-benchmark datasets)
    elif data_args.train_data.endswith(".fa"):
        if model_args.train_task.lower() == "regression":
            labels = None
        else:
            # create id map
            id2label = {idx: label for idx, label in enumerate(labels)}
            label2id = {label: idx for idx, label in enumerate(labels)}
            print(label2id)

        train_dic = load_fasta_data(data_args.train_data, labels, data_type='train', sample=data_args.samples, seed=training_args.seed)
        eval_dic = load_fasta_data(data_args.eval_data, labels, data_type='dev', sample=int(data_args.samples*data_args.split), seed=training_args.seed)
        test_dic = load_fasta_data(data_args.test_data, labels, data_type='test', sample=int(data_args.samples*data_args.split), seed=training_args.seed)
        train_ds = Dataset.from_dict(train_dic, split="train")
        if eval_dic and test_dic:
            test_ds = Dataset.from_dict(test_dic, split="test")
            eval_ds = Dataset.from_dict(eval_dic, split="dev")
            dataset = DatasetDict({'train': train_ds, 'test': test_ds, 'dev': eval_ds})
        elif eval_dic:
            eval_ds = Dataset.from_dict(eval_dic, split="dev")
            dataset = DatasetDict({'train': train_ds, 'dev': eval_ds})
        elif test_dic:
            test_ds = Dataset.from_dict(test_dic, split="test")
            dataset = DatasetDict({'train': train_ds, 'test': test_ds})
        else:
            dataset = DatasetDict({'train': train_ds})

    # check the training task
    if model_args.train_task.lower().endswith("classification"):
        num_labels = len(labels)
    elif model_args.train_task.lower().startswith("multi-label"):
        num_labels = len(labels)
    elif model_args.train_task.lower() == "regression":
        num_labels = 1
    else:
        num_labels = 0

    # pre-processing the input data
    def encode(example):
        if data_args.kmer > 0:
            sequence = seq2kmer(example['sequence'], data_args.kmer)
            tokenized_seq = tokenizer(sequence, truncation=True, padding='max_length',
                                      max_length=training_args.model_max_length)
            return tokenized_seq
        else:
            return tokenizer([x.upper() for x in example['sequence']],
                             truncation=True, padding='max_length',
                             max_length=training_args.model_max_length)

    dataset = dataset.map(encode, batched=True)
    # dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # load model
    if num_labels:
        if model_args.train_task.lower().startswith("multi-label"):
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                problem_type="multi_label_classification",
                trust_remote_code=True
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=num_labels,
                trust_remote_code=True,
                # ignore_mismatched_sizes=True
            )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
        )

    # in case token_id is different
    model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.vocab_size < len(tokenizer):
        print('Update tokenizer length:', len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

    # define computer_metrics
    compute_metrics = evaluate_metrics(model_args.train_task.lower())

    # define trainer
    if ('DNABERT-2' in model_args.model_name_or_path) or ('dnabert2' in model_args.model_name_or_path.lower()):
        from pdllib.metrics import metrics_for_dnabert2
        compute_metrics, preprocess_logits_for_metrics = metrics_for_dnabert2(model_args.train_task)

        trainer = Trainer(model=model, tokenizer=tokenizer,
                          args=training_args,
                          train_dataset=dataset['train'],
                          eval_dataset=dataset['dev'] if 'dev' in dataset else dataset['test'],
                          compute_metrics=compute_metrics,
                          preprocess_logits_for_metrics=preprocess_logits_for_metrics)
    else:
        trainer = Trainer(model=model, tokenizer=tokenizer,
                          args=training_args,
                          train_dataset=dataset['train'],
                          eval_dataset=dataset['dev'] if 'dev' in dataset else dataset['test'],
                          compute_metrics=compute_metrics)

    if model_args.load_checkpoint:
        trainer.train(model_args.load_checkpoint)
    else:
        trainer.train()

    # save model in hf format
    trainer.save_model(training_args.output_dir)
    # model.save_pretrained(training_args.output_dir)

    # do prediction if test datasets is existed
    if 'test' in dataset and 'dev' in dataset:
        model.eval()
        results = trainer.predict(dataset['test'])
        with open(training_args.output_dir + "/test_metrics.json", "w") as outf:
            print(results.metrics, file=outf)


if __name__ == "__main__":
    train()
