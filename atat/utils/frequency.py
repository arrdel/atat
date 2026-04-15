"""
Token Frequency Computation (§3.1)

Computes per-token frequency statistics from the training corpus for
the frequency component of the hybrid importance estimator:

    i^l_freq = 1 - log(freq(x^l) + 1) / log(max_freq + 1)

Rare tokens → high frequency importance, common tokens → low.

Usage:
    python -m atat.utils.frequency --data-dir ./data_cache --output freq_table.pt
"""

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_frequency_table(
    tokenizer,
    texts,
    vocab_size: int = 50257,
    max_seq_len: int = 1024,
) -> torch.Tensor:
    """
    Compute raw token frequency counts from a corpus.

    Args:
        tokenizer:   GPT-2 tokenizer.
        texts:       Iterable of text strings.
        vocab_size:  Vocabulary size (50257).
        max_seq_len: Maximum sequence length for tokenization.

    Returns:
        counts: (V,) tensor of integer frequency counts.
    """
    counts = torch.zeros(vocab_size, dtype=torch.long)

    for text in tqdm(texts, desc="Computing token frequencies"):
        if not text or len(text.strip()) < 10:
            continue
        tokens = tokenizer(
            text,
            max_length=max_seq_len,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        for token_id in tokens.tolist():
            if 0 <= token_id < vocab_size:
                counts[token_id] += 1

    return counts


def frequency_importance(
    input_ids: torch.Tensor,
    frequency_table: torch.Tensor,
) -> torch.Tensor:
    """
    Compute frequency-based importance scores (Eq. 2):

        i^l_freq = 1 - log(freq(x^l) + 1) / log(max_freq + 1)

    Args:
        input_ids:       (B, L) token ids.
        frequency_table: (V,) raw frequency counts.

    Returns:
        importance: (B, L) in [0, 1].
    """
    freqs = frequency_table[input_ids.long()]  # (B, L)
    max_freq = frequency_table.max().float()
    importance = 1.0 - torch.log(freqs.float() + 1) / torch.log(max_freq + 1)
    return importance.clamp(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Compute token frequency table")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data_cache",
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="freq_table.pt",
        help="Output path for frequency table",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openwebtext",
        help="Dataset name",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit number of documents",
    )
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    logger.info(f"Loading {args.dataset}...")
    ds = load_dataset(args.dataset, split="train", cache_dir=args.data_dir)

    texts = (item["text"] for item in ds)
    if args.max_docs:
        import itertools
        texts = itertools.islice(texts, args.max_docs)

    counts = compute_frequency_table(tokenizer, texts)

    torch.save(counts, args.output)
    logger.info(f"Saved frequency table to {args.output}")
    logger.info(f"  Total tokens: {counts.sum().item():,}")
    logger.info(f"  Unique tokens: {(counts > 0).sum().item():,}")
    logger.info(f"  Max frequency: {counts.max().item():,}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
