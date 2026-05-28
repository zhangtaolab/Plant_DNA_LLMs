# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import os
from tqdm import tqdm
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    mean_squared_error, r2_score, hamming_loss
)
from mindspeed_llm import megatron_adaptor
from megatron.training import get_args, print_rank_0, get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.models.mamba import Mamba2ForSequenceClassification
from megatron.core.transformer.spec_utils import import_module
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.initialize import initialize_megatron
from megatron.training.training import setup_model_and_optimizer
from megatron.training.checkpointing import save_checkpoint

class CustomCSVDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['sequence']) 
        label = self.df.iloc[idx]['label']
        inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'position_ids': torch.arange(inputs['input_ids'].size(1), dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_dl(path, tokenizer, max_seq_len, shuffle=True):
    args = get_args()
    ds = CustomCSVDataset(path, tokenizer, max_seq_len)
    sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=mpu.get_data_parallel_world_size(),
        rank=mpu.get_data_parallel_rank(), shuffle=shuffle
    )
    return DataLoader(ds, batch_size=args.micro_batch_size, sampler=sampler), sampler
    
def model_provider(pre_process=True, post_process=True):
    args = get_args()
    config = core_transformer_config_from_args(args)
    mamba_stack_spec = import_module(args.spec)
    model = Mamba2ForSequenceClassification(
        config=config, mamba_stack_spec=mamba_stack_spec,
        vocab_size=args.padded_vocab_size, max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process, post_process=post_process, parallel_output=True,
        num_labels=args.num_labels, problem_type=args.problem_type,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
    )
    return model

def add_extra_args(parser):
    group = parser.add_argument_group(title='genomic_finetune')
    group.add_argument('--num_labels', type=int, default=3)
    group.add_argument('--problem_type', type=str, default='single_label_classification')
    group.add_argument('--train_csv', type=str, required=True)
    group.add_argument('--dev_csv', type=str, required=True)
    group.add_argument('--test_csv', type=str, required=True)
    group.add_argument('--tensorboard_dir', type=str, default='tensorboard_logs')
    group.add_argument('--epochs', type=int, default=3)
    group.add_argument('--log_interval', type=int, default=10)
    return parser

@torch.no_grad()
def evaluate(model, dataloader, writer, global_step, desc="Eval"):
    args = get_args()
    model.eval()
    all_preds, all_labels = [], []
    total_eval_loss = 0
    device = torch.npu.current_device() 

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        output = model(input_ids=input_ids, labels=labels, 
                       attention_mask=batch['attention_mask'].to(device),
                       position_ids=batch['position_ids'].to(device))
        
        loss = output['loss'] if isinstance(output, dict) else output
        logits = output['logits'] if isinstance(output, dict) else None
        total_eval_loss += loss.item()
        
        if args.problem_type == "regression":
            preds = logits.squeeze(-1)
        elif args.problem_type == "multi_label_classification":
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
        else:
            preds = torch.argmax(logits, dim=-1)
            
        all_preds.append(preds.float())
        all_labels.append(labels.float())

    # 分布式汇总
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        gathered_preds = [torch.zeros_like(all_preds) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(all_labels) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_preds, all_preds)
        torch.distributed.all_gather(gathered_labels, all_labels)
        full_preds = torch.cat(gathered_preds, dim=0).cpu().numpy()
        full_labels = torch.cat(gathered_labels, dim=0).cpu().numpy()
    else:
        full_preds = all_preds.cpu().numpy()
        full_labels = all_labels.cpu().numpy()

    if mpu.get_data_parallel_rank() == 0:
        avg_loss = total_eval_loss / len(dataloader)
        metrics_results = {"Loss": avg_loss}

        if args.problem_type == "regression":
            metrics_results.update({
                "MSE": mean_squared_error(full_labels, full_preds),
                "R2": r2_score(full_labels, full_preds)
            })
        elif args.problem_type == "multi_label_classification":
            h_loss = hamming_loss(full_labels, full_preds)
            f1 = f1_score(full_labels, full_preds, average='samples', zero_division=0)
            metrics_results.update({"HammingLoss": h_loss, "F1_Samples": f1})
        else:
            # 单标签分类：应用要求的使用 weighted 平均的指标
            acc = accuracy_score(full_labels, full_preds)
            f1 = f1_score(full_labels, full_preds, average='weighted', zero_division=0)
            prec = precision_score(full_labels, full_preds, average='weighted', zero_division=0)
            rec = recall_score(full_labels, full_preds, average='weighted', zero_division=0)
            metrics_results.update({"Acc": acc, "F1": f1, "Precision": prec, "Recall": rec})
            
        print_rank_0(f">>> {desc} | Loss: {avg_loss:.4f} | F1(weighted): {f1:.4f}")

        if writer:
            for name, value in metrics_results.items():
                writer.add_scalar(f'{desc}/{name}', value, global_step)

    model.train()
    return avg_loss

def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()

    # 自动处理分类头尺寸不匹配逻辑（保持不变）
    original_load = nn.Module.load_state_dict
    def patched_load(self, state_dict, strict=True):
        if 'output_layer.weight' in state_dict and hasattr(self, 'output_layer'):
            if state_dict['output_layer.weight'].shape != self.output_layer.weight.shape:
                state_dict.pop('output_layer.weight', None)
                state_dict.pop('output_layer.bias', None)
        return original_load(self, state_dict, strict=False)
    nn.Module.load_state_dict = patched_load

    tokenizer = get_tokenizer().tokenizer
    device = torch.npu.current_device()

    writer = None
    if mpu.get_data_parallel_rank() == 0:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.tensorboard_dir)

    model_list, optimizer, _ = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)
    model = model_list[0]

    train_loader, train_sampler = create_dl(args.train_csv, tokenizer, args.seq_length)
    dev_loader, _ = create_dl(args.dev_csv, tokenizer, args.seq_length, shuffle=False)
    test_loader, _ = create_dl(args.test_csv, tokenizer, args.seq_length, shuffle=False)

    print_rank_0(">>> Starting Fine-tuning with tqdm...")
    global_step = 0
    total_steps = len(train_loader) * args.epochs
    
    ### 【新增】初始化最优指标和路径
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.save, "best_model_weights.pt")
    os.makedirs(args.save, exist_ok=True)

    pbar = None
    if mpu.get_data_parallel_rank() == 0:
        pbar = tqdm(total=total_steps, desc="Finetuning", unit="it", dynamic_ncols=True)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device).long()
            
            output = model(input_ids=input_ids, labels=labels, 
                           attention_mask=batch['attention_mask'].to(device),
                           position_ids=batch['position_ids'].to(device))
            
            loss = output['loss'] if isinstance(output, dict) else output
            loss.backward()
            optimizer.step()
            global_step += 1

            if writer and global_step % args.log_interval == 0:
                writer.add_scalar('Train/Loss', loss.item(), global_step)

            if pbar:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "epoch": epoch})

            # --- 间隔验证 ---
            if global_step % args.eval_interval == 0:
                if pbar: pbar.write(f">>> Interval Eval at Step {global_step}")
                ### 【修改】接收返回值并判断最优
                val_loss = evaluate(model, dev_loader, writer, global_step, desc="Eval_Step")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if mpu.get_data_parallel_rank() == 0:
                        torch.save(model.state_dict(), best_model_path)
                        if pbar: pbar.write(f"*** Best Model Saved (Loss: {val_loss:.4f}) ***")

            # --- 定期 Checkpoint ---
            if global_step % args.save_interval == 0:
                if pbar: pbar.write(f">>> Saving Checkpoint at Step {global_step}")
                save_checkpoint(global_step, model_list, optimizer, None, 0)

        # --- Epoch 结束验证 ---
        ### 【修改】接收返回值并判断最优
        val_loss = evaluate(model, dev_loader, writer, global_step, desc=f"Eval_Epoch")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if mpu.get_data_parallel_rank() == 0:
                torch.save(model.state_dict(), best_model_path)
                print_rank_0(f"*** Best Model Saved at Epoch End (Loss: {val_loss:.4f}) ***")

    if pbar: pbar.close()
    
    ###测试前加载最优权重
    if os.path.exists(best_model_path):
        print_rank_0(f">>> Loading best weights from {best_model_path} for testing...")
        # map_location 确保在分布式环境下加载到正确的卡上
        best_state = torch.load(best_model_path, map_location=f"npu:{device}")
        model.load_state_dict(best_state)
    
    print_rank_0(">>> Starting Final Test Set Evaluation...")
    evaluate(model, test_loader, writer, global_step, desc="Final_Test")

    if writer: writer.close()
    print_rank_0(">>> All Tasks Completed.")

if __name__ == "__main__":
    main()