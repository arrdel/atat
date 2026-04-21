#!/usr/bin/env python3
"""
Prepare tokenized OpenWebText cache for SEDD baseline training.

Creates pre-tokenized train/val splits saved with HuggingFace save_to_disk(),
which SEDD's TextDataset expects via load_from_disk().

Usage:
    python atat/scripts/data/prepare_openwebtext_cache.py
"""

import os
from pathlib import Path
from datasets import load_dataset
from transformers import GPT2TokenizerFast


def main():
    cache_dir = Path(__file__).parent.parent.parent / "experiments" / "generalizability" / "data_cache"
    train_path = cache_dir / "openwebtext_train"
    val_path = cache_dir / "openwebtext_val"

    if train_path.exists() and val_path.exists():
        print("Cache already exists, skipping.")
        return

    os.makedirs(cache_dir, exist_ok=True)

    print("Loading OpenWebText...")
    dataset = load_dataset(
        "Skylion007/openwebtext",
        trust_remote_code=True,
        num_proc=8,
        download_mode="reuse_cache_if_exists",
    )

    # Create train/val split (use 1% for val)
    split = dataset["train"].train_test_split(test_size=0.01, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]

    print(f"Train: {len(train_ds):,} examples, Val: {len(val_ds):,} examples")

    # Tokenize
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=1024,
            padding=False,
        )

    print("Tokenizing train split (single process for stability)...")
    train_ds = train_ds.map(tokenize, batched=True, batch_size=5000, num_proc=1, remove_columns=["text"])
    print("Tokenizing val split...")
    val_ds = val_ds.map(tokenize, batched=True, batch_size=5000, num_proc=1, remove_columns=["text"])

    print(f"Saving to {cache_dir}...")
    train_ds.save_to_disk(str(train_path))
    val_ds.save_to_disk(str(val_path))

    print("✓ Done!")


if __name__ == "__main__":
    main()
