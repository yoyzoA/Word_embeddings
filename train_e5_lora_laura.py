#!/usr/bin/env python3
"""
Train an E5 embedding model with LoRA using a grouped "LAURA-style" paraphrase collapse loss.

Input dataset (row-wise JSONL), one JSON per line:
{
  "base_phrase_id": 17,
  "base_phrase": "the market fell today",
  "variant": "the market dropped today"
}

Multiple rows share the same base_phrase_id.

Run:
  pip install -U torch transformers peft accelerate
  python train_e5_lora_laura.py --data ptb_pairs.jsonl --out out/e5_lora --epochs 1 --batch 32 --lr 2e-4 --max_len 256 --bf16
"""
# python train_e5_lora_laura.py \
#   --data ptb_pairs.jsonl \
#   --out out/e5_lora \
#   --epochs 1 \
#   --batch 64 \
#   --grad_accum 4 \
#   --lr 2e-4 \
#   --max_len 128 \
#   --fp16

import argparse
import json
import os
from collections import defaultdict
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType


# ----------------------------
# Dataset: row-wise JSONL
# ----------------------------
class PairRowJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.rows: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {ln}: {e}") from e

                # Required keys
                for k in ("base_phrase_id", "base_phrase", "variant"):
                    if k not in obj:
                        raise ValueError(f"Missing key '{k}' on line {ln}")

                self.rows.append(obj)

        if not self.rows:
            raise ValueError("Dataset is empty.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        # E5 likes prefixes; keep consistent for both sides
        return {
            "id": int(r["base_phrase_id"]),
            "base": f"query: {str(r['base_phrase'])}",
            "variant": f"query: {str(r['variant'])}",
        }


# ----------------------------
# Collator
# ----------------------------
class PairRowCollator:
    def __init__(self, tokenizer, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        ids = [x["id"] for x in batch]
        bases = [x["base"] for x in batch]
        variants = [x["variant"] for x in batch]

        tb = self.tokenizer(
            bases,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        tv = self.tokenizer(
            variants,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "id": ids,  # keep as python list for grouping
            "base_ids": tb["input_ids"],
            "base_mask": tb["attention_mask"],
            "var_ids": tv["input_ids"],
            "var_mask": tv["attention_mask"],
        }


# ----------------------------
# Embedding helpers (mean pooling + L2 norm)
# ----------------------------
def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden: [B, T, H], attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def encode(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    emb = mean_pool(out.last_hidden_state, attention_mask)
    emb = F.normalize(emb, p=2, dim=-1)
    return emb


# ----------------------------
# Grouped LAURA loss (mean within group, mean over groups)
# ----------------------------
def grouped_laura_loss(base_emb: torch.Tensor, var_emb: torch.Tensor, ids: List[int]) -> torch.Tensor:
    """
    base_emb: [B, H] embeddings for base phrases (row-wise, may repeat per id)
    var_emb:  [B, H] embeddings for variants (aligned row-wise)
    ids:      list length B, base_phrase_id per row

    For each unique id in the batch:
      - pick v_hat as the base embedding from the first row of that id
      - take all variant embeddings for that id
      - L(i) = mean_j (1 - cosine(v_hat, v_ij))
    Global batch loss = mean_i L(i) over unique ids.
    """
    groups = defaultdict(list)
    for row_idx, gid in enumerate(ids):
        groups[gid].append(row_idx)

    losses = []
    for gid, idxs in groups.items():
        # Anchor: base embedding from first occurrence of this id in the batch
        v_hat = base_emb[idxs[0]]  # [H]
        v_j = var_emb[idxs]        # [N_i, H]

        # cosine distance: 1 - cos
        d = 1.0 - (v_hat.unsqueeze(0) * v_j).sum(dim=1)  # [N_i]
        losses.append(d.mean())

    return torch.stack(losses).mean()


# ----------------------------
# Custom Trainer
# ----------------------------
class LauraTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        ids = inputs["id"]

        base_ids = inputs["base_ids"].to(model.device)
        base_mask = inputs["base_mask"].to(model.device)
        var_ids = inputs["var_ids"].to(model.device)
        var_mask = inputs["var_mask"].to(model.device)

        base_emb = encode(model, base_ids, base_mask)
        var_emb = encode(model, var_ids, var_mask)

        loss = grouped_laura_loss(base_emb, var_emb, ids)

        return (loss, {"base_emb": base_emb, "var_emb": var_emb}) if return_outputs else loss


# ----------------------------
# LoRA setup
# ----------------------------
def build_lora_e5(model_name: str, r: int, alpha: int, dropout: float, target_modules: List[str]):
    base_model = AutoModel.from_pretrained(model_name)

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to row-wise JSONL dataset")
    ap.add_argument("--model", default="intfloat/e5-base-v2", help="HF model name")
    ap.add_argument("--out", default="./out_e5_lora_laura", help="Output directory")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)

    # precision
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")

    # LoRA params
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="query,key,value,dense",
                    help="Comma-separated module names for LoRA injection")

    # training niceties
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=50)

    args = ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    model = build_lora_e5(
        model_name=args.model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    train_ds = PairRowJsonlDataset(args.data)
    collator = PairRowCollator(tokenizer, max_len=args.max_len)

    train_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        optim="adamw_torch",
    )

    trainer = LauraTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    trainer.train()

    # Save LoRA adapter + tokenizer
    trainer.model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    print(f"\n✅ Done. Saved LoRA adapter + tokenizer to: {args.out}")
    print("Tip: Load with AutoModel + PeftModel.from_pretrained(...) to apply the adapter.")


if __name__ == "__main__":
    main()
