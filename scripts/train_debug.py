import os, sys, math, pathlib, yaml, torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim import AdamW

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.data.packed_ds import NpzPackedDataset, simple_collate
from src.utils.utils import load_yaml
import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config", default="configs/model/model_debug.json")
    ap.add_argument("--train_config", default="configs/train/debug.yaml")
    args = ap.parse_args()
    return args

def main():
    args = parse_args()
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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model configuration
    model_cfg = AutoConfig.from_pretrained(args.model_config)

    # Load model
    model = AutoModelForCausalLM.from_config(model_cfg)
    if use_grad_ckpt:
        model.gradient_checkpointing_enable()
    model = model.to(device)
    model.train()

    # Load dataset
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
        betas=betas,
        weight_decay=weight_decay,
    )
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps
    )
    scaler = torch.amp.GradScaler(enabled=use_amp and not use_bf16)
    autocast_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_amp else None)

    os.makedirs(save_dir, exist_ok=True)
    it = iter(dl)

    for step in range(1, max_steps + 1):
        batch = next(it)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        if autocast_dtype is not None:
            with torch.amp.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=autocast_dtype):
                out = model(**batch, labels=batch["input_ids"])
                loss = out.loss / max(1, gradient_accumulation_steps)
        else:
            out = model(**batch, labels=batch["input_ids"])
            loss = out.loss / max(1, gradient_accumulation_steps)
        
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if step % gradient_accumulation_steps == 0:
            if scaler.is_enabled():
                scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            sched.step()
            model.zero_grad(set_to_none=True)
        
        if step % log_steps == 0:
            tok_per_step = batch["input_ids"].numel() * max(1, train_micro_batch_size_per_gpu)
            print(f"step {step:5d} | loss {loss.item()*max(1,gradient_accumulation_steps):.4f} | tokens/step {tok_per_step}")
        
        if step % save_steps == 0 or step == max_steps:
            ckpt_dir = os.path.join(save_dir, f"step_{step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            print(f"saved -> {ckpt_dir}")
        
    print("Training complete.")

if __name__ == "__main__":
    main()