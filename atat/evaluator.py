"""
ATAT Evaluator (Appendix §D.2, Table 10)

NELBO-based perplexity evaluation following MDLM's protocol.

"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


BENCHMARKS = {
    "wikitext2": {
        "hf_name": "wikitext",
        "hf_config": "wikitext-2-raw-v1",
        "split": "test",
        "text_key": "text",
        "domain": "Wikipedia",
    },
    "lambada": {
        "hf_name": "lambada",
        "hf_config": None,
        "split": "test",
        "text_key": "text",
        "domain": "Fiction/narrative",
    },
    "ptb": {
        "hf_name": "ptb_text_only",
        "hf_config": None,
        "split": "test",
        "text_key": "sentence",
        "domain": "WSJ news",
    },
    "lm1b": {
        "hf_name": "lm1b",
        "hf_config": None,
        "split": "test",
        "text_key": "text",
        "domain": "Web (shuffled)",
    },
    "agnews": {
        "hf_name": "ag_news",
        "hf_config": None,
        "split": "test",
        "text_key": "text",
        "domain": "News articles",
    },
    "pubmed": {
        "hf_name": "ccdv/pubmed-summarization",
        "hf_config": None,
        "split": "test",
        "text_key": "article",
        "domain": "Biomedical",
    },
    "arxiv": {
        "hf_name": "ccdv/arxiv-summarization",
        "hf_config": None,
        "split": "test",
        "text_key": "article",
        "domain": "Scientific",
    },
}


class ATATEvaluator:
    """
    NELBO-based perplexity evaluator for ATAT.

    Args:
        model:          ATATDiT model.
        tokenizer:      GPT-2 tokenizer.
        nfe:            Number of function evaluations (1000).
        seq_len:        Maximum sequence length (1024).
        mask_token_id:  Mask token id (50256).
        seed:           Random seed (42).
        n_eval_runs:    Number of evaluation runs for averaging (3).
    """

    def __init__(
        self,
        model,
        tokenizer,
        nfe: int = 1000,
        seq_len: int = 1024,
        mask_token_id: int = 50256,
        seed: int = 42,
        n_eval_runs: int = 3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.nfe = nfe
        self.seq_len = seq_len
        self.mask_token_id = mask_token_id
        self.seed = seed
        self.n_eval_runs = n_eval_runs

        # Log-linear noise schedule: α_t = 1 - t
        self.timesteps = torch.linspace(0, 1, nfe + 1)

    def _noise_schedule(self, t: float) -> float:
        """Log-linear noise schedule: α_t = 1 - t."""
        return 1.0 - t

    @torch.no_grad()
    def compute_nelbo(
        self,
        input_ids: torch.Tensor,
        frequency_table: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute NELBO bound for a batch of sequences.

        The NELBO for absorbing diffusion is:
            NELBO = Σ_t  (α_t - α_{t-Δ}) / (1 - α_t) · E_{x_t} [-log p_θ(x_0|x_t)]

        where x_t is sampled by masking tokens with probability 1 - α_t.

        Args:
            input_ids: (B, L) clean token ids.
            frequency_table: Optional frequency table for importance.

        Returns:
            Average NELBO per token (in nats).
        """
        self.model.eval()
        B, L = input_ids.shape
        device = input_ids.device

        total_nelbo = 0.0

        for step_idx in range(1, self.nfe + 1):
            t_curr = self.timesteps[step_idx].item()
            t_prev = self.timesteps[step_idx - 1].item()

            alpha_curr = self._noise_schedule(t_curr)
            alpha_prev = self._noise_schedule(t_prev)

            # Mask rate at time t
            mask_rate = 1.0 - alpha_curr

            if mask_rate < 1e-8:
                continue

            # Sample x_t by masking each token with probability mask_rate
            mask_probs = torch.full((B, L), mask_rate, device=device)
            mask_decisions = torch.bernoulli(mask_probs).bool()
            x_t = input_ids.clone()
            x_t[mask_decisions] = self.mask_token_id

            # Get importance for conditioning
            importance = self.model.get_importance(
                input_ids, frequency_table=frequency_table
            )

            # Forward pass
            t_batch = torch.full((B,), t_curr, device=device)
            logits = self.model(x_t, t_batch, importance)

            # Cross-entropy loss at masked positions only
            log_probs = F.log_softmax(logits, dim=-1)  # (B, L, V)
            target_log_probs = log_probs.gather(
                2, input_ids.unsqueeze(-1)
            ).squeeze(-1)  # (B, L)

            # Only count loss at masked positions
            masked_loss = -target_log_probs * mask_decisions.float()

            # NELBO weight: (α_curr - α_prev) / (1 - α_curr)
            weight = (alpha_curr - alpha_prev) / max(1.0 - alpha_curr, 1e-8)

            # Sum over positions, average over batch
            step_nelbo = (masked_loss.sum(dim=-1) * weight).mean()
            total_nelbo += step_nelbo.item()

        # Average per token
        nelbo_per_token = total_nelbo / L
        return nelbo_per_token

    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        dataset_name: str = "",
        max_batches: Optional[int] = None,
        frequency_table: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Evaluate NELBO-based PPL on a full dataset.

        Args:
            dataloader:      DataLoader yielding {"input_ids": (B, L)}.
            dataset_name:    Name for logging.
            max_batches:     Limit number of batches.
            frequency_table: Optional frequency table.

        Returns:
            Dictionary with PPL, NELBO, and standard deviation.
        """
        all_nelbos = []

        for run_idx in range(self.n_eval_runs):
            torch.manual_seed(self.seed + run_idx)
            run_nelbos = []

            for batch_idx, batch in enumerate(tqdm(
                dataloader,
                desc=f"{dataset_name} run {run_idx+1}/{self.n_eval_runs}",
            )):
                if max_batches and batch_idx >= max_batches:
                    break

                input_ids = batch["input_ids"].cuda()
                nelbo = self.compute_nelbo(input_ids, frequency_table)
                run_nelbos.append(nelbo)

            run_avg = np.mean(run_nelbos) if run_nelbos else 0.0
            all_nelbos.append(run_avg)

        mean_nelbo = np.mean(all_nelbos)
        std_nelbo = np.std(all_nelbos)
        ppl = math.exp(mean_nelbo)
        ppl_std = ppl * std_nelbo  # first-order approximation

        result = {
            "ppl": ppl,
            "ppl_std": ppl_std,
            "nelbo": mean_nelbo,
            "nelbo_std": std_nelbo,
            "n_runs": self.n_eval_runs,
            "nfe": self.nfe,
        }

        logger.info(
            f"{dataset_name:>12s}: PPL = {ppl:.2f} ± {ppl_std:.2f}"
            f" (NELBO = {mean_nelbo:.4f} ± {std_nelbo:.4f})"
        )

        return result

    def evaluate_all_benchmarks(
        self,
        cache_dir: str = "./data_cache",
        batch_size: int = 4,
        max_batches: Optional[int] = None,
        frequency_table: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate on all 7 benchmarks from the paper.

        Args:
            cache_dir:       HuggingFace cache directory.
            batch_size:      Evaluation batch size.
            max_batches:     Limit batches per dataset.
            frequency_table: Optional frequency table.

        Returns:
            Dictionary mapping benchmark name → results.
        """
        from datasets import load_dataset
        from torch.utils.data import Dataset

        class TokenizedDataset(Dataset):
            def __init__(self, texts, tokenizer, max_len):
                self.encodings = []
                for text in texts:
                    if not text or len(text.strip()) < 10:
                        continue
                    enc = tokenizer(
                        text,
                        max_length=max_len,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    self.encodings.append(enc["input_ids"].squeeze(0))

            def __len__(self):
                return len(self.encodings)

            def __getitem__(self, idx):
                return {"input_ids": self.encodings[idx]}

        results = {}

        for name, cfg in BENCHMARKS.items():
            logger.info(f"Evaluating on {name} ({cfg['domain']})...")

            try:
                if cfg["hf_config"]:
                    ds = load_dataset(
                        cfg["hf_name"],
                        cfg["hf_config"],
                        split=cfg["split"],
                        cache_dir=cache_dir,
                    )
                else:
                    ds = load_dataset(
                        cfg["hf_name"],
                        split=cfg["split"],
                        cache_dir=cache_dir,
                    )

                texts = [item[cfg["text_key"]] for item in ds]
                tok_ds = TokenizedDataset(texts, self.tokenizer, self.seq_len)

                if len(tok_ds) == 0:
                    logger.warning(f"No valid sequences for {name}, skipping")
                    continue

                loader = DataLoader(tok_ds, batch_size=batch_size, shuffle=False)
                results[name] = self.evaluate_dataset(
                    loader,
                    dataset_name=name,
                    max_batches=max_batches,
                    frequency_table=frequency_table,
                )

            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
                continue

        # Print summary table
        print("\n" + "=" * 70)
        print("ATAT Evaluation Results (NELBO-based PPL, 1000 NFE)")
        print("=" * 70)
        print(f"{'Benchmark':>12s}  {'PPL':>8s}  {'± std':>8s}  {'Domain'}")
        print("-" * 70)
        for name, res in results.items():
            domain = BENCHMARKS[name]["domain"]
            print(
                f"{name:>12s}  {res['ppl']:8.2f}  ±{res['ppl_std']:7.2f}  {domain}"
            )
        print("=" * 70)

        return results
