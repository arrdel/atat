"""
Packed Sequence DataLoader (Appendix §D.1)

Loads and packs sequences for training, following MDLM's protocol:
    - Tokenize with GPT-2 BPE tokenizer (K = 50,257)
    - Concatenate documents with <EOS> separators
    - Pack into chunks of length L = 1024
    - No padding during training

Usage:
    from atat.utils.dataloader import create_train_dataloader
    loader = create_train_dataloader(cache_dir="./data_cache", batch_size=64)
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger(__name__)


class PackedSequenceDataset(Dataset):
    """
    Dataset that packs tokenized documents into fixed-length chunks.

    Documents are concatenated with EOS separators and then split
    into non-overlapping chunks of length `seq_len`.

    Args:
        token_ids:  1-D tensor of all token ids (pre-packed).
        seq_len:    Chunk length (1024).
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int = 1024):
        self.seq_len = seq_len
        # Drop last incomplete chunk
        n_chunks = len(token_ids) // seq_len
        self.token_ids = token_ids[: n_chunks * seq_len].reshape(n_chunks, seq_len)

    def __len__(self):
        return self.token_ids.shape[0]

    def __getitem__(self, idx):
        return {"input_ids": self.token_ids[idx]}


def tokenize_and_pack(
    dataset,
    tokenizer,
    text_key: str = "text",
    seq_len: int = 1024,
    eos_token_id: int = 50256,
    max_docs: Optional[int] = None,
) -> torch.Tensor:
    """
    Tokenize a HuggingFace dataset and pack into a single 1-D tensor.

    Documents are separated by EOS tokens.

    Args:
        dataset:       HuggingFace dataset.
        tokenizer:     GPT-2 tokenizer.
        text_key:      Column name for text.
        seq_len:       Sequence length (for logging only).
        eos_token_id:  EOS token id (50256).
        max_docs:      Limit number of documents.

    Returns:
        1-D tensor of all token ids.
    """
    all_ids = []
    n_docs = 0

    for item in dataset:
        text = item.get(text_key, "")
        if not text or len(text.strip()) < 10:
            continue

        tokens = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)
        all_ids.extend(tokens.tolist())
        all_ids.append(eos_token_id)

        n_docs += 1
        if max_docs and n_docs >= max_docs:
            break

    token_ids = torch.tensor(all_ids, dtype=torch.long)
    n_chunks = len(token_ids) // seq_len

    logger.info(
        f"Packed {n_docs:,} documents → {len(token_ids):,} tokens"
        f" → {n_chunks:,} chunks of length {seq_len}"
    )

    return token_ids


def create_train_dataloader(
    cache_dir: str = "./data_cache",
    dataset_name: str = "openwebtext",
    batch_size: int = 64,
    seq_len: int = 1024,
    num_workers: int = 4,
    distributed: bool = False,
    max_docs: Optional[int] = None,
) -> DataLoader:
    """
    Create training DataLoader with packed sequences.

    Args:
        cache_dir:    HuggingFace cache directory.
        dataset_name: Dataset to load (default: openwebtext).
        batch_size:   Batch size (64).
        seq_len:      Sequence length (1024).
        num_workers:  DataLoader workers.
        distributed:  Use DistributedSampler for DDP.
        max_docs:     Limit number of documents.

    Returns:
        DataLoader yielding {"input_ids": (B, L)} batches.
    """
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    logger.info(f"Loading {dataset_name} from {cache_dir}...")
    ds = load_dataset(dataset_name, split="train", cache_dir=cache_dir)

    token_ids = tokenize_and_pack(
        ds, tokenizer, seq_len=seq_len, max_docs=max_docs
    )

    dataset = PackedSequenceDataset(token_ids, seq_len=seq_len)

    sampler = DistributedSampler(dataset) if distributed else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def create_eval_dataloader(
    dataset_name: str,
    hf_name: str,
    hf_config: Optional[str] = None,
    split: str = "test",
    text_key: str = "text",
    cache_dir: str = "./data_cache",
    batch_size: int = 4,
    seq_len: int = 1024,
    num_workers: int = 2,
) -> DataLoader:
    """
    Create evaluation DataLoader for a benchmark dataset.

    Args:
        dataset_name: Display name.
        hf_name:      HuggingFace dataset identifier.
        hf_config:    HuggingFace dataset config (optional).
        split:        Dataset split.
        text_key:     Column name for text.
        cache_dir:    HuggingFace cache directory.
        batch_size:   Batch size.
        seq_len:      Sequence length.
        num_workers:  DataLoader workers.

    Returns:
        DataLoader yielding {"input_ids": (B, L)} batches.
    """
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    if hf_config:
        ds = load_dataset(hf_name, hf_config, split=split, cache_dir=cache_dir)
    else:
        ds = load_dataset(hf_name, split=split, cache_dir=cache_dir)

    token_ids = tokenize_and_pack(
        ds, tokenizer, text_key=text_key, seq_len=seq_len
    )

    dataset = PackedSequenceDataset(token_ids, seq_len=seq_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
