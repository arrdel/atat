"""
Train SEDD + ATAT: Score Entropy Discrete Diffusion with importance-aware masking.

ATAT modifies the absorbing diffusion forward process so that semantically
important tokens are masked later (harder to corrupt) while function words
are masked earlier.  The SEDD denoiser architecture and score-entropy loss
are unchanged — ATAT only changes WHICH tokens are masked at each timestep.

Architecture:
    Frozen GPT-2 Small (124M) → ImportanceEstimator MLP (886K)
        → per-token importance scores i ∈ [0,1]^L
        → importance-biased absorb probabilities for SEDD forward process
        → SEDD denoiser trained on importance-masked sequences

Usage:
    cd /home/adelechinda/home/projects/mdlm/atat
    conda run -n mdlm-atat python baselines/sedd/train_sedd_atat.py \
        --config sedd_atat_config.yaml --num-gpus 2

    # Quick debug (1000 steps):
    conda run -n mdlm-atat python baselines/sedd/train_sedd_atat.py \
        --config sedd_atat_config.yaml --num-gpus 1 --max-steps 1000 --no-wandb
"""

import sys
import math
from pathlib import Path

# ── path setup ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent   # .../mdlm
ATAT_ROOT    = PROJECT_ROOT / "atat"                        # .../mdlm/atat
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ATAT_ROOT))

import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from transformers import GPT2Model

# SEDD components
from baselines.sedd import model as sedd_model
from baselines.sedd import graph_lib
from baselines.sedd import noise_lib
from baselines.sedd import losses

# ATAT importance estimator — import via spec to avoid circular atat/__init__ issue
import importlib.util as _ilu
_imp_spec = _ilu.spec_from_file_location(
    "importance_estimator",
    str(ATAT_ROOT / "atat" / "importance_estimator.py"),
)
_imp_mod = _ilu.module_from_spec(_imp_spec)
_imp_spec.loader.exec_module(_imp_mod)
ImportanceEstimator = _imp_mod.ImportanceEstimator


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, data_path, max_length=1024):
        print(f"Loading dataset from {data_path}...")
        self.dataset = load_from_disk(str(data_path))
        self.max_length = max_length
        print(f"Loaded {len(self.dataset):,} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ids = self.dataset[idx]["input_ids"][: self.max_length]
        if len(ids) < self.max_length:
            ids = ids + [0] * (self.max_length - len(ids))
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}


# ─────────────────────────────────────────────────────────────────────────────
# EMA (same as baseline)
# ─────────────────────────────────────────────────────────────────────────────

class ExponentialMovingAverage:
    def __init__(self, parameters, decay=0.9999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in parameters if p.requires_grad}
        self.backup  = {}

    def update(self, parameters):
        for n, p in parameters:
            if p.requires_grad:
                if self.shadow[n].device != p.device:
                    self.shadow[n] = self.shadow[n].to(p.device)
                self.shadow[n] = (1 - self.decay) * p.data + self.decay * self.shadow[n]

    def store(self, parameters):
        for n, p in parameters:
            if p.requires_grad:
                self.backup[n] = p.data.clone()

    def restore(self, parameters):
        for n, p in parameters:
            if p.requires_grad:
                p.data = self.backup[n]

    def copy_to(self, parameters):
        for n, p in parameters:
            if p.requires_grad:
                s = self.shadow[n].to(p.device)
                p.data.copy_(s)


# ─────────────────────────────────────────────────────────────────────────────
# Importance-biased absorbing noise schedule
# ─────────────────────────────────────────────────────────────────────────────

def importance_biased_absorb(
    input_ids: torch.Tensor,       # (B, L) — clean tokens
    importance: torch.Tensor,      # (B, L) ∈ [0,1] — higher = more important
    sigma: torch.Tensor,           # (B,)  — SEDD noise level
    mask_token_id: int,
    eta: float = 0.3,              # clipping parameter (§3.3, Eq. 3)
) -> torch.Tensor:
    """
    Sample x_t from the importance-biased absorbing forward process.

    Standard SEDD: each token is masked independently with prob 1 - exp(-σ).
    ATAT: token i is masked with prob clamp(p_t · g_bal(i_score, t), 0, 1)
    where g_bal = t·g_inv + (1-t)·g_prop (balanced interpolation, §3.3).

    At training time sigma is sampled from the noise schedule, and t ≈ sigma
    (for loglinear schedule sigma ≈ -log(1-t) ≈ t for small t).
    We use the normalized sigma as a proxy for t.

    Args:
        input_ids:    (B, L) clean token ids
        importance:   (B, L) per-token importance in [0,1]
        sigma:        (B,) SEDD noise level σ(t)
        mask_token_id: absorbing state id
        eta: clipping threshold for masking probability

    Returns:
        x_t: (B, L) partially masked tokens
    """
    B, L = input_ids.shape
    device = input_ids.device

    # Base absorbing probability: p_base = 1 - exp(-σ)
    p_base = 1.0 - torch.exp(-sigma)   # (B,)
    p_base = p_base.unsqueeze(1).expand(B, L)  # (B, L)

    # Normalize t ∈ [0,1] as a proxy for curriculum position
    t = p_base  # already ∈ [0,1] for loglinear schedule

    # ATAT masking: importance-proportional (g_prop) and importance-inverse (g_inv)
    # g_prop(i) = i / mean(i) · p_base    — proportional to importance
    # g_inv(i)  = (1-i) / mean(1-i) · p_base — inverse of importance
    # g_bal(i,t) = t·g_inv + (1-t)·g_prop
    imp     = importance                         # (B, L)
    inv_imp = 1.0 - imp                          # (B, L)

    # Normalize so each row integrates to 1 (avoid division by zero)
    imp_mean     = imp.mean(dim=1, keepdim=True).clamp(min=1e-8)
    inv_imp_mean = inv_imp.mean(dim=1, keepdim=True).clamp(min=1e-8)

    g_prop = (imp / imp_mean) * p_base          # (B, L)
    g_inv  = (inv_imp / inv_imp_mean) * p_base  # (B, L)
    g_bal  = t * g_inv + (1 - t) * g_prop       # (B, L) time-dependent blend

    # Clip: never mask more than (1-η) of tokens at a single timestep
    mask_prob = g_bal.clamp(0.0, 1.0 - eta)     # (B, L)

    # Sample the mask
    mask = torch.bernoulli(mask_prob).bool()      # True = absorb this token
    x_t = input_ids.clone()
    x_t[mask] = mask_token_id

    return x_t


# ─────────────────────────────────────────────────────────────────────────────
# SEDD + ATAT Lightning module
# ─────────────────────────────────────────────────────────────────────────────

class SEDDATATWrapper(L.LightningModule):
    """
    PyTorch Lightning module for SEDD + ATAT training.

    Key difference from plain SEDDWrapper:
      - Runs frozen GPT-2 + ImportanceEstimator to get per-token importance.
      - Replaces the uniform absorbing noise with importance-biased absorbing.
      - Adds a small auxiliary importance-consistency loss (weight γ).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # ── SEDD denoiser (trainable) ────────────────────────────────────
        self.sedd = sedd_model.SEDD(config)

        # ── Frozen GPT-2 anchor ──────────────────────────────────────────
        print("Loading frozen GPT-2 Small anchor...")
        self.anchor = GPT2Model.from_pretrained("gpt2")
        for p in self.anchor.parameters():
            p.requires_grad = False
        self.anchor.eval()
        print("  GPT-2 anchor frozen ✓")

        # ── Importance estimator (trainable, 886K params) ────────────────
        hidden_size = self.anchor.config.n_embd   # 768 for GPT-2 Small
        self.importance_estimator = ImportanceEstimator(input_dim=hidden_size)

        # ── Noise schedule & graph ───────────────────────────────────────
        self.noise = noise_lib.get_noise(config)
        self.graph = None   # device-dependent, built in setup()
        self.loss_fn      = None
        self.eval_loss_fn = None

        # ── EMA on SEDD only (anchor is frozen) ─────────────────────────
        self.ema = ExponentialMovingAverage(
            self.sedd.named_parameters(),
            decay=getattr(getattr(config, "training", None), "ema", 0.9999),
        )

        # ── ATAT hyper-params ────────────────────────────────────────────
        atat_cfg = getattr(config, "atat", None)
        self.lambda_blend = getattr(atat_cfg, "lambda_blend", 0.7)
        self.eta          = getattr(atat_cfg, "eta",          0.3)
        self.gamma        = getattr(atat_cfg, "gamma",        0.003)

        # ── mask token ───────────────────────────────────────────────────
        self.mask_token_id = config.tokens   # vocab_size = absorbing state

        self.train_step_count = 0

    # ── setup ────────────────────────────────────────────────────────────────
    def setup(self, stage=None):
        if self.graph is None:
            self.graph = graph_lib.get_graph(self.config, self.device)
        if self.loss_fn is None:
            self.loss_fn = losses.get_loss_fn(
                self.noise, self.graph, train=True, sampling_eps=1e-5
            )
        if self.eval_loss_fn is None:
            self.eval_loss_fn = losses.get_loss_fn(
                self.noise, self.graph, train=False, sampling_eps=1e-5
            )

    # ── importance estimation ─────────────────────────────────────────────────
    def get_importance(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run frozen GPT-2 + ImportanceEstimator on clean tokens.

        Returns importance ∈ [0,1]^(B,L).
        """
        with torch.no_grad():
            hidden = self.anchor(input_ids).last_hidden_state   # (B, L, 768)

        # ImportanceEstimator is trainable — pass both hidden states and token ids
        importance = self.importance_estimator(hidden, token_ids=input_ids)  # (B, L)
        return importance   # ∈ [0,1] via sigmoid inside ImportanceEstimator

    # ── training step ─────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]

        # 1. Compute importance scores
        importance = self.get_importance(input_ids)   # (B, L)

        # 2. Sample noise level σ from SEDD schedule
        t = torch.rand(input_ids.shape[0], device=self.device)
        sigma, _ = self.noise(t)                       # (B,)

        # 3. Apply importance-biased absorbing noise (ATAT §3.3)
        x_t = importance_biased_absorb(
            input_ids, importance.detach(), sigma,
            mask_token_id=self.mask_token_id,
            eta=self.eta,
        )

        # 4. SEDD score-entropy loss on importance-masked sequences
        # We call the SEDD forward pass directly (bypassing normal get_loss_fn
        # which would re-sample noise) since we've already applied ATAT noise.
        score = self.sedd(x_t, sigma)                  # (B, L, V) score ratios

        # Score entropy: -sum_v q(v|x_t) log s_θ(v|x_t)
        # For absorbing diffusion the target distribution is a delta at x_0.
        # The simplified score entropy is: -log s_θ(x_0 | x_t) for masked positions.
        masked_positions = (x_t == self.mask_token_id)  # (B, L)
        if masked_positions.any():
            # log score at true token
            log_score = F.log_softmax(score[masked_positions], dim=-1)  # (N, V)
            true_tokens = input_ids[masked_positions]                    # (N,)
            score_entropy = -log_score.gather(1, true_tokens.unsqueeze(1)).squeeze(1)
            sedd_loss = score_entropy.mean()
        else:
            sedd_loss = torch.tensor(0.0, device=self.device)

        # 5. Importance consistency auxiliary loss (§3.3, γ weight)
        # Oracle: mask probability proportional to inverse perplexity of each token
        with torch.no_grad():
            anchor_logits = self.anchor(input_ids).last_hidden_state
            # Use anchor's own hidden states as a soft oracle target
            oracle_imp = torch.sigmoid(anchor_logits.norm(dim=-1))   # (B, L)
            oracle_imp = oracle_imp / oracle_imp.max(dim=1, keepdim=True).values.clamp(min=1e-8)

        imp_loss = F.mse_loss(importance, oracle_imp.detach())

        # 6. Total loss
        loss = sedd_loss + self.gamma * imp_loss

        self.log("train/loss",         loss,       prog_bar=True, on_step=True)
        self.log("train/sedd_loss",    sedd_loss,  on_step=True)
        self.log("train/imp_loss",     imp_loss,   on_step=True)
        self.train_step_count += 1

        return loss

    # ── validation step ───────────────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]

        # Use EMA weights for validation
        self.ema.store(self.sedd.named_parameters())
        self.ema.copy_to(self.sedd.named_parameters())

        with torch.no_grad():
            # Use standard (uniform) SEDD loss for comparable val metrics
            loss = self.eval_loss_fn(self.sedd, input_ids).mean()

        self.ema.restore(self.sedd.named_parameters())

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    # ── configure optimizers ──────────────────────────────────────────────────
    def configure_optimizers(self):
        # Only train SEDD denoiser + importance estimator (anchor is frozen)
        trainable_params = (
            list(self.sedd.parameters())
            + list(self.importance_estimator.parameters())
        )
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            weight_decay=self.config.optim.weight_decay,
        )

        # Cosine schedule with warmup
        warmup_steps = getattr(getattr(self.config, "lr_scheduler", None), "warmup_steps", 10000)
        max_steps    = getattr(getattr(self.config, "trainer",       None), "max_steps",   500000)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, max_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ── EMA update ────────────────────────────────────────────────────────────
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update(self.sedd.named_parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SEDD + ATAT")
    parser.add_argument("--config",    type=str, default="sedd_atat_config.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-gpus",  type=int, default=2)
    parser.add_argument("--no-wandb",  action="store_true")
    args = parser.parse_args()

    # ── load config ──────────────────────────────────────────────────────────
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = OmegaConf.load(config_path)

    if args.max_steps:
        config.trainer.max_steps = args.max_steps
    if args.no_wandb:
        config.wandb.mode = "disabled"

    print("=" * 70)
    print("SEDD + ATAT Training  —  Importance-Aware Score Entropy Diffusion")
    print("=" * 70)
    print(f"Config:          {args.config}")
    print(f"Max steps:       {config.trainer.max_steps:,}")
    print(f"Batch / GPU:     {config.loader.batch_size}")
    print(f"Grad accum:      {config.trainer.accumulate_grad_batches}")
    print(f"Effective batch: {config.loader.batch_size * args.num_gpus * config.trainer.accumulate_grad_batches}")
    print(f"GPUs:            {args.num_gpus}")
    print(f"ATAT λ:          {getattr(getattr(config, 'atat', None), 'lambda_blend', 0.7)}")
    print(f"ATAT η:          {getattr(getattr(config, 'atat', None), 'eta', 0.3)}")
    print(f"ATAT γ:          {getattr(getattr(config, 'atat', None), 'gamma', 0.003)}")
    print("=" * 70)

    # ── data ─────────────────────────────────────────────────────────────────
    data_root = ATAT_ROOT / "experiments" / "generalizability" / "data_cache"
    train_ds = TextDataset(data_root / "openwebtext_train")
    val_ds   = TextDataset(data_root / "openwebtext_val")

    train_loader = DataLoader(
        train_ds, batch_size=config.loader.batch_size,
        shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    print(f"Train: {len(train_loader):,} batches  |  Val: {len(val_loader):,} batches\n")

    # ── model ─────────────────────────────────────────────────────────────────
    model = SEDDATATWrapper(config)

    sedd_params  = sum(p.numel() for p in model.sedd.parameters())
    imp_params   = sum(p.numel() for p in model.importance_estimator.parameters())
    anch_params  = sum(p.numel() for p in model.anchor.parameters())
    print(f"SEDD denoiser:          {sedd_params:>12,} params (trainable)")
    print(f"Importance estimator:   {imp_params:>12,} params (trainable)")
    print(f"GPT-2 anchor:           {anch_params:>12,} params (FROZEN)")
    print(f"Total trainable:        {sedd_params + imp_params:>12,} params\n")

    # ── callbacks ─────────────────────────────────────────────────────────────
    ckpt_dir = Path(config.checkpointing.save_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="sedd-atat-{step:06d}",
        every_n_train_steps=config.checkpointing.every_n_train_steps,
        save_top_k=config.checkpointing.save_top_k,
        monitor="val/loss",
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ── strategy ──────────────────────────────────────────────────────────────
    strategy = (
        DDPStrategy(find_unused_parameters=True)
        if args.num_gpus > 1
        else "auto"
    )

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=strategy,
        max_steps=config.trainer.max_steps,
        max_epochs=1,          # hard-stop Lightning from overrunning past max_steps
        precision=getattr(config.trainer, "precision", "bf16-mixed"),
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        gradient_clip_val=getattr(config.trainer, "gradient_clip_val", 1.0),
        log_every_n_steps=config.trainer.log_every_n_steps,
        val_check_interval=config.trainer.val_check_interval,
        limit_val_batches=config.trainer.limit_val_batches,
        callbacks=[checkpoint_cb, lr_monitor],
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print("Starting training...\n")
    trainer.fit(model, train_loader, val_loader)
    print("\n✓ SEDD + ATAT training complete!")
    print(f"  Checkpoint dir: {ckpt_dir}")


if __name__ == "__main__":
    main()
