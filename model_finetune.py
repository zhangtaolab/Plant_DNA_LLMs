# coding=utf-8
#! /usr/bin/env python

import os
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_IB_DISABLE'] = '1'

import random
import numpy as np
import json
import logging
from collections import namedtuple
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Sequence, Tuple, List
from tqdm import tqdm

from scipy.special import softmax
import sklearn
import evaluate
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, \
                         TrainingArguments, Trainer, HfArgumentParser
from datasets import Dataset, DatasetDict, load_dataset

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf


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
    evaluation_strategy: str = field(default='epoch'),
    # eval_accumulation_steps: Optional[int] = field(default=None),
    warmup_steps: int = field(default=50)
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


############################################################
## Mamba model configurations ##############################

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
            ClassificationOutput = namedtuple("ClassificationOutput", ["logits"])
            return ClassificationOutput(logits=logits)
        else:
            ClassificationOutput = namedtuple("ClassificationOutput", ["loss", "logits"])

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        return ClassificationOutput(loss=loss, logits=logits)

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

        print("Newly initialized embedding:", set(model.state_dict().keys()) - set(model_state_dict.keys()))
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
            ClassificationOutput = namedtuple("ClassificationOutput", ["logits"])
            return ClassificationOutput(logits=logits)
        else:
            ClassificationOutput = namedtuple("ClassificationOutput", ["loss", "logits"])

        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())

        return ClassificationOutput(loss=loss, logits=logits)

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

        print("Newly initialized embedding:", set(model.state_dict().keys()) - set(model_state_dict.keys()))
        return model


class MambaTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        labels = inputs.pop('labels')

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir = None, _internal_call = False):
        if output_dir is None:
            output_dir = self.args.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")

        self.tokenizer.save_pretrained(output_dir)

        with open(f'{output_dir}/config.json', 'w') as f:
            json.dump(self.model.config.to_dict(), f)

############################################################


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
        kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
        kmers = " ".join(kmer)
        all_kmers.append(kmers)
    return all_kmers


# Define evaluation metrics
def calculate_metric_with_sklearn(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    print(valid_labels.shape, valid_predictions.shape)
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }


def evaluate_metrics(task):
    if task == "regression":
        # mse_metric = evaluate.load("evaluate/metrics/mse/mse.py")
        r2_metric = evaluate.load("evaluate/metrics/r_squared/r_squared.py")
        spm_metric = evaluate.load("evaluate/metrics/spearmanr/spearmanr.py")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred

            r2 = r2_metric.compute(references=labels, predictions=logits)
            spearman = spm_metric.compute(references=labels, predictions=logits)

            return {"r2": r2, "spearmanr": spearman['spearmanr']}

    elif task == "multi-classification":
        metric1 = evaluate.load("evaluate/metrics/precision/precision.py")
        metric2 = evaluate.load("evaluate/metrics/recall/recall.py")
        metric3 = evaluate.load("evaluate/metrics/f1/f1.py")
        metric4 = evaluate.load("evaluate/metrics/matthews_correlation/matthews_correlation.py")
        roc_metric = evaluate.load("evaluate/metrics/roc_auc/roc_auc.py", "multiclass")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            logits = logits[0] if isinstance(logits, tuple) else logits
            # predictions = np.argmax(logits, axis=-1)
            pred_probs = softmax(logits, axis=1)
            predictions = [x.tolist().index(max(x)) for x in pred_probs]

            precision = metric1.compute(predictions=predictions, references=labels, average="micro")
            recall = metric2.compute(predictions=predictions, references=labels, average="micro")
            f1 = metric3.compute(predictions=predictions, references=labels, average="micro")
            mcc = metric4.compute(predictions=predictions, references=labels)
            roc_auc_ovr = roc_metric.compute(references=labels,
                                            prediction_scores=pred_probs,
                                            multi_class='ovr')
            roc_auc_ovo = roc_metric.compute(references=labels,
                                            prediction_scores=pred_probs,
                                            multi_class='ovo')

            return {**precision, **recall, **f1, **mcc, "AUROC_ovr": roc_auc_ovr['roc_auc'], "AUROC_ovo": roc_auc_ovo['roc_auc']}

    else:
        clf_metrics = evaluate.combine(["evaluate/metrics/accuracy/accuracy.py",
                                        "evaluate/metrics/f1/f1.py",
                                        "evaluate/metrics/precision/precision.py",
                                        "evaluate/metrics/recall/recall.py",
                                        "evaluate/metrics/matthews_correlation/matthews_correlation.py"])

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            logits = logits[0] if isinstance(logits, tuple) else logits
            predictions = np.argmax(logits, axis=-1)
            return clf_metrics.compute(predictions=predictions, references=labels)

    return compute_metrics


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # define model source
    if os.path.exists(model_args.model_name_or_path):
        model_args.source = "local"
        prefix = ""
    else:
        if model_args.source == "huggingface":
            prefix = "https://huggingface.co/"
        elif model_args.source == "modelscope":
            prefix = "https://modelscope.cn/models/"
        else:
            prefix = "https://huggingface.co/"
    model_args.model_name_or_path = prefix + model_args.model_name_or_path

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
            return tokenizer(example['sequence'], truncation=True, padding='max_length',
                             max_length=training_args.model_max_length)

    dataset = dataset.map(encode, batched=True)
    # dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # load model
    if num_labels:
        if 'mamba' in model_args.model_name_or_path.lower():
            if num_labels == 1:
                model = MambaSequenceRegression.from_pretrained(model_args.model_name_or_path)
            else:
                model = MambaSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                                    num_classes=num_labels)
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
        print(len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

    # define computer_metrics
    compute_metrics = evaluate_metrics(model_args.train_task.lower())

    # define trainer
    if ('DNABERT-2' in model_args.model_name_or_path) or ('dnabert2' in model_args.model_name_or_path.lower()):
        r2_metric = evaluate.load("evaluate/metrics/r_squared/r_squared.py")
        spm_metric = evaluate.load("evaluate/metrics/spearmanr/spearmanr.py")
        clf_metrics = evaluate.combine(["evaluate/metrics/accuracy/accuracy.py",
                                        "evaluate/metrics/f1/f1.py",
                                        "evaluate/metrics/precision/precision.py",
                                        "evaluate/metrics/recall/recall.py",
                                        "evaluate/metrics/matthews_correlation/matthews_correlation.py"])
        metric1 = evaluate.load("evaluate/metrics/precision/precision.py")
        metric2 = evaluate.load("evaluate/metrics/recall/recall.py")
        metric3 = evaluate.load("evaluate/metrics/f1/f1.py")
        metric4 = evaluate.load("evaluate/metrics/matthews_correlation/matthews_correlation.py")
        roc_metric = evaluate.load("evaluate/metrics/roc_auc/roc_auc.py", "multiclass")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            # print(labels.shape, logits[0].shape)
            if model_args.train_task.lower() == "regression":
                r2 = r2_metric.compute(references=labels, predictions=logits[0])
                spearman = spm_metric.compute(references=labels, predictions=logits[0])
                return {"r2": r2, "spearmanr": spearman['spearmanr']}
            else:
                if model_args.train_task.lower() == "classification":
                    predictions = torch.argmax(torch.from_numpy(logits[0]), dim=-1)
                    return clf_metrics.compute(predictions=predictions, references=labels)
                else:
                    pred_probs = softmax(logits[0], axis=1)
                    predictions = [x.tolist().index(max(x)) for x in pred_probs]
                    precision = metric1.compute(predictions=predictions, references=labels, average="micro")
                    recall = metric2.compute(predictions=predictions, references=labels, average="micro")
                    f1 = metric3.compute(predictions=predictions, references=labels, average="micro")
                    mcc = metric4.compute(predictions=predictions, references=labels)
                    roc_auc_ovr = roc_metric.compute(references=labels,
                                            prediction_scores=pred_probs,
                                            multi_class='ovr')
                    roc_auc_ovo = roc_metric.compute(references=labels,
                                            prediction_scores=pred_probs,
                                            multi_class='ovo')
                    return {**precision, **recall, **f1, **mcc, "AUROC_ovr": roc_auc_ovr['roc_auc'], "AUROC_ovo": roc_auc_ovo['roc_auc']}

        def preprocess_logits_for_metrics(logits, labels):
            """
            Original Trainer may have a memory leak.
            This is a workaround to avoid storing too many tensors that are not needed.
            """
            logits = logits[0] if isinstance(logits, tuple) else logits
            # pred_ids = torch.argmax(logits, dim=-1)
            return logits, labels

        trainer = Trainer(model=model, tokenizer=tokenizer,
                          args=training_args,
                          train_dataset=dataset['train'],
                          eval_dataset=dataset['dev'] if 'dev' in dataset else dataset['test'],
                          compute_metrics=compute_metrics,
                          preprocess_logits_for_metrics=preprocess_logits_for_metrics)
    elif 'mamba' in model_args.model_name_or_path.lower():
        trainer = MambaTrainer(
                  model=model,
                  tokenizer=tokenizer,
                  args=training_args,
                  train_dataset=dataset["train"],
                  eval_dataset=dataset['dev'] if 'dev' in dataset else dataset['test'],
                  compute_metrics=compute_metrics)
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
