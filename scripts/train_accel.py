"""
Train Qwen3-0.6B (or compatible) with Accelerate + DeepSpeed.

Usage (3x4090 example):
  accelerate launch \
    --num_processes 3 \
    --gpu_ids 0,1,2 \
    --deepspeed configs/deepseed/zero3_bf16.json \
    scripts/train_accel.py \
    --model_config configs/model/qwen3_0.6b.json \
    --train_config configs/train/6B_tokens.yaml \
    --tokenizer_dir tokenizer
"""

import os, sys, math, pathlib, json, yaml, torch
from dataclasses import dataclass
from typing import Optional, Dict, Any

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
from accelerate import Accelerator
from torch.optim import AdamW

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.data.packed_ds import NpzPackedDataset, simple_collate
from src.utils.utils import load_yaml, bf16_supported
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config", default="configs/model/qwen3_0.6b.json", help="Path to model configuration file.")
    ap.add_argument("--train_config", default="configs/train/6B_tokens.yaml", help="Path to training configuration file.")
    ap.add_argument("--tokenizer_dir", default="tokenizer", help="Directory containing the tokenizer.")
    ap.add_argument("--resume", default=None, help="Path to a checkpoint to resume training from.")
    args = ap.parse_args()

    cfg = load_yaml(args.train_config)

    # Load training configuration
    data_dir = cfg["data_dir"]
    train_micro_batch_size_per_gpu = cfg["train_micro_batch_size_per_gpu"]
    gradient_accumulation_steps = cfg["gradient_accumulation_steps"]
    pin_memory = cfg["pin_memory"]
    num_workers = cfg["num_workers"]
    prefetch_factor = cfg["prefetch_factor"]
    presistent_workers = cfg["persistent_workers"]
    lr = cfg["learning_rate"]
    betas = cfg["betas"]
    weight_decay = cfg["weight_decay"]
    max_steps = cfg["max_steps"]
    warmup_steps = cfg["warmup_steps"]
    use_bf16 = cfg["bf16"]
    use_amp = cfg["amp"]
    use_grad_ckpt = cfg["gradient_checkpointing"]
    save_steps = cfg["save_interval"]
    eval_steps = cfg["eval_interval"]
    log_steps = cfg["logging_interval"]
    save_dir = cfg["save_dir"]
    seed = cfg["seed"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    mp = "bf16" if (use_bf16 and bf16_supported()) else "no"
    accelerator = Accelerator(
        mixed_precision=mp,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    device = accelerator.device
    if accelerator.is_main_process:
        print(f"[accelerate] mixed_precision={mp} world_size={accelerator.num_processes}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=False)
    if args.resume:
        if accelerator.is_main_process:
            print(f"[resume] loading model from {args.resume}")
        model = AutoModelForCausalLM.from_pretrained(
            args.resume,
            torch_dtype=torch.bfloat16 if mp == "bf16" else None,
        )
    else:
        model_cfg = AutoConfig.from_pretrained(args.model_config)
        if getattr(model_cfg, "vocab_size", None) != len(tokenizer):
            if accelerator.is_main_process:
                print(f"[info] adjust vocab_size: {getattr(model_cfg,'vocab_size', None)} -> {len(tokenizer)}")
            model_cfg.vocab_size = len(tokenizer)
        model = AutoModelForCausalLM.from_config(model_cfg)
        if use_grad_ckpt:
            model.gradient_checkpointing_enable()
        
    model = model.to(device)
    model.train()

    ds = NpzPackedDataset(data_dir, shuffle_files=True, seed=seed)
    dl = DataLoader(
        ds,
        batch_size=train_micro_batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=simple_collate,
        prefetch_factor=prefetch_factor,
        persistent_workers=presistent_workers,
    )

    optim = AdamW(
        model.parameters(),
        lr=lr,
        betas=tuple(betas),
        weight_decay=weight_decay,
    )
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps
    )

    model, optim, dl, sched = accelerator.prepare(model, optim, dl, sched)

    os.makedirs(save_dir, exist_ok=True)
    running = 0.0
    data_iter = iter(dl)

    for step in tqdm(range(1, max_steps + 1), total=max_steps, desc="Training", dynamic_ncols=True):
        batch = next(data_iter)
        with accelerator.accumulate(model):
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)

        running += loss.item()

        if step % log_steps == 0 and accelerator.is_main_process:
            dp = accelerator.num_processes
            toks = train_micro_batch_size_per_gpu * batch["input_ids"].shape[1] * dp
            print(f"step {step:7d} | loss {running/log_steps:.4f} | tokens/step ~ {toks}")
            running = 0.0
        
        if (step % save_steps == 0 or step == max_steps) and accelerator.is_main_process:
            save_path = os.path.join(save_dir, f"step-{step:07d}")
            state_dict = accelerator.get_state_dict(model)
            accelerator.unwrap_model(model).save_pretrained(
                save_path,
                state_dict=state_dict,
                safe_serialization=True,
                )
            
            tokenizer.save_pretrained(save_path)
            torch.save(
                {
                "optimizer": optim.state_dict(),
                "scheduler": sched.state_dict(),
                "step": step,
                },
                os.path.join(save_path, "trainer_state.pt"),
            )
            print(f"[save] model and tokenizer saved to {save_path}")
    
    if accelerator.is_main_process:
        print("Training complete.")

if __name__ == "__main__":
    main()