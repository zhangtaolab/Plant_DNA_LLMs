# coding=utf-8
#! /usr/bin/env python

import os
# Uncomment this if you want to use specific directory for cache
# os.environ['HF_HOME'] = '/mnt/data'
# Uncomment this to use the accelerate mirror for Hugging Face downloads
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, HfArgumentParser


## Arguments ###############################################

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="./models/plantdnabert")
    tokenizer_path: Optional[str] = field(default=None)
    load_checkpoint: Optional[str] = field(default=None)
    is_mlm: bool = field(default=False, metadata={"help": "Is masked language model."})

@dataclass
class DataArguments:
    train_data: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data: Optional[str] = field(default=None, metadata={"help": "Path to the valid data."})
    test_data: Optional[str] = field(default=None, metadata={"help": "Path to the test data."})
    split: float = field(default=0.1, metadata={"help": "Test split"})
    shuffle: bool = field(default=True)

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
    logging_steps: Optional[int] = field(default=100)
    logging_strategy: str = field(default='steps'),
    save_steps: Optional[int] = field(default=500)
    save_strategy: str = field(default='steps'),
    # eval_accumulation_steps: Optional[int] = field(default=None),
    # warmup_steps: int = field(default=50)
    warmup_ratio: float = field(default=0.05)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.98)
    adam_epsilon:float = field(default=1e-6)
    save_total_limit: int = field(default=10)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    seed: int = field(default=7)
    no_safe_serialization: bool = field(default=False)

############################################################


def train(args, model, trainset, validset, data_collator, output_dir, resume_from_checkpoint=None):
    
    trainer = Trainer(
        args=args,
        model=model,
        train_dataset=trainset,
        eval_dataset=validset,
        data_collator=data_collator
    )

    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_model(output_dir)


def main():
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if model_args.is_mlm:
        # Masked Language Model
        model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    else:
        # Causal Language Model
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    output_dir = training_args.output_dir

    def encode(examples):
        return tokenizer(examples["text"],
                         truncation=True, padding='max_length',
                         max_length=training_args.model_max_length)
    
    data_sets = load_dataset('text', data_files=data_args.train_data, split='train')
    data_sets = data_sets.train_test_split(test_size=data_args.split, seed=training_args.seed, shuffle=data_args.shuffle)
    data_sets = data_sets.map(encode, batched=True)
    trainset, validset = data_sets['train'], data_sets['test']
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=model_args.is_mlm)
    
    train(training_args, model, trainset, validset, data_collator, output_dir, resume_from_checkpoint=model_args.load_checkpoint)


if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        sys.stderr.write("User interrupt\n")
        sys.exit(0)