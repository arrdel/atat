"""
Simple training script for AR Transformer baseline.
Standalone version that creates its own dataloaders.
"""

import os
import sys
from pathlib import Path

# Add baselines to path
BASELINES_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASELINES_DIR))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from datasets import load_from_disk
import argparse
from tqdm import tqdm

from ar_transformer.model import ARTransformer


class TextDataset(Dataset):
    """Simple dataset for tokenized text."""
    
    def __init__(self, data_path, max_length=1024, max_examples=None):
        print(f"Loading dataset from {data_path}...")
        self.dataset = load_from_disk(data_path)
        if max_examples is not None:
            print(f"Limiting to first {max_examples} examples for testing")
            self.dataset = self.dataset.select(range(min(max_examples, len(self.dataset))))
        self.max_length = max_length
        print(f"Loaded {len(self.dataset)} examples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = item['input_ids'][:self.max_length]
        
        # Pad if necessary
        if len(input_ids) < self.max_length:
            input_ids = input_ids + [0] * (self.max_length - len(input_ids))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long)
        }


class ARTransformerLightning(L.LightningModule):
    """PyTorch Lightning wrapper for AR Transformer."""
    
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Create model
        self.model = ARTransformer(
            vocab_size=config['vocab_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            ffn_dim=config['ffn_dim'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout'],
        )
        
        print(f"Model initialized with {self.model.num_parameters():,} parameters")
    
    def forward(self, input_ids):
        return self.model(input_ids)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        input_ids = batch["input_ids"]
        loss, metrics = self.model.compute_loss(input_ids)
        
        # Log metrics
        self.log("train/loss", metrics["loss"], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/perplexity", metrics["perplexity"], on_step=True, on_epoch=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids = batch["input_ids"]
        loss, metrics = self.model.compute_loss(input_ids)
        
        # Log metrics
        self.log("val/loss", metrics["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/perplexity", metrics["perplexity"], on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters with and without weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to biases and layer norms
            if "bias" in name or "ln" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.config['weight_decay']},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config['learning_rate'],
            betas=self.config['betas'],
            eps=self.config['eps'],
        )
        
        # Create learning rate scheduler
        def lr_lambda(current_step):
            """Cosine decay with warmup."""
            warmup_steps = self.config['warmup_steps']
            max_steps = self.config['max_steps']
            min_lr_ratio = self.config['min_lr'] / self.config['learning_rate']
            
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                import math
                return max(min_lr_ratio, 0.5 * (1.0 + math.cos(progress * math.pi)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train AR Transformer baseline")
    parser.add_argument("--data-path", type=str, 
                       default="./data_cache/openwebtext",
                       help="Path to preprocessed data")
    parser.add_argument("--max-steps", type=int, default=500000, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--accumulate", type=int, default=32, help="Gradient accumulation steps")
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--val-every", type=int, default=5000, help="Validate every N steps")
    parser.add_argument("--save-every", type=int, default=10000, help="Save checkpoint every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit dataset size for testing")
    args = parser.parse_args()
    
    # Config
    config = {
        'vocab_size': 50257,
        'hidden_size': 768,
        'num_layers': 12,
        'num_heads': 12,
        'ffn_dim': 3072,
        'max_seq_len': args.max_length,
        'dropout': 0.1,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'betas': [0.9, 0.999],
        'eps': 1e-8,
        'warmup_steps': 10000,
        'max_steps': args.max_steps,
        'min_lr': 1e-5,
    }
    
    # Set seed
    L.seed_everything(42, workers=True)
    
    # Create datasets
    print("Loading data...")
    # Handle different data path formats
    if "wrapped.dat" in args.data_path or args.data_path.endswith("-train") or args.data_path.endswith("_train"):
        # Path already includes train suffix or is wrapped format
        train_path = args.data_path
        # Replace train with valid/validation - handle specific patterns
        if "openwebtext-train_train_bs1024_wrapped.dat" in train_path:
            val_path = train_path.replace("openwebtext-train_train_bs1024_wrapped.dat", 
                                         "openwebtext-valid_validation_bs1024_wrapped.dat")
        elif "-train_train" in train_path:
            val_path = train_path.replace("-train_train", "-valid_validation")
        elif "-train" in train_path:
            val_path = train_path.replace("-train", "-valid")
        elif "_train" in train_path:
            val_path = train_path.replace("_train", "_validation")
        else:
            val_path = train_path.replace("train", "valid")
    else:
        # Add train/valid suffixes
        train_path = f"{args.data_path}-train"
        val_path = f"{args.data_path}-valid"
    
    print(f"Train path: {train_path}")
    print(f"Valid path: {val_path}")
    
    train_dataset = TextDataset(train_path, max_length=args.max_length, max_examples=args.max_examples)
    val_dataset = TextDataset(val_path, max_length=args.max_length, max_examples=100 if args.max_examples else None)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,  # Smaller for validation
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = ARTransformerLightning(config)
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_dir = "./outputs/baselines/ar_transformer"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="ar-{step:06d}-{val/perplexity:.2f}",
        monitor="val/perplexity",
        mode="min",
        save_top_k=3,
        every_n_train_steps=args.save_every,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Create logger
    logger = None
    if not args.no_wandb:
        logger = WandbLogger(
            project="mdlm-baselines",
            name="ar-transformer-baseline",
            tags=["baseline", "autoregressive", "phase2"],
            config=config,
            offline=True,  # Use offline mode
        )
    
    # Adjust val_check_interval for small datasets
    val_check_interval = min(args.val_every, len(train_loader))
    
    # Create trainer
    trainer = L.Trainer(
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=args.num_gpus,
        strategy="ddp" if args.num_gpus > 1 else "auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate,
        callbacks=callbacks,
        logger=logger,
        val_check_interval=val_check_interval,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train
    print("Starting training...")
    print(f"Effective batch size: {args.batch_size} * {args.num_gpus} * {args.accumulate} = {args.batch_size * args.num_gpus * args.accumulate}")
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )
    
    print("Training complete!")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Best val PPL: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
