"""
Training script for D3PM using PyTorch Lightning.

Usage:
    python train_d3pm.py --max-steps 500000 --num-gpus 2
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.d3pm.d3pm_model import D3PM
from baselines.d3pm.diffusion import GaussianDiffusion


class TextDataset(Dataset):
    """Simple dataset for tokenized text."""
    
    def __init__(self, data_path, max_length=1024):
        print(f"Loading dataset from {data_path}...")
        self.dataset = load_from_disk(data_path)
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


class D3PMWrapper(pl.LightningModule):
    """PyTorch Lightning wrapper for D3PM training."""
    
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Create model and diffusion
        self.model = D3PM(config)
        self.diffusion = GaussianDiffusion(config)
        
        # Training config
        if hasattr(config, 'training'):
            train_config = config.training
            self.learning_rate = float(train_config.learning_rate) if hasattr(train_config, 'learning_rate') else 3e-4
            self.weight_decay = float(train_config.weight_decay) if hasattr(train_config, 'weight_decay') else 0.01
            self.warmup_steps = int(train_config.warmup_steps) if hasattr(train_config, 'warmup_steps') else 10000
            self.max_steps = int(train_config.max_steps) if hasattr(train_config, 'max_steps') else 500000
        else:
            self.learning_rate = float(config.get('learning_rate', 3e-4))
            self.weight_decay = float(config.get('weight_decay', 0.01))
            self.warmup_steps = int(config.get('warmup_steps', 10000))
            self.max_steps = int(config.get('max_steps', 500000))
    
    def forward(self, x, t):
        return self.model(x, t)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Get tokens from batch
        if isinstance(batch, dict):
            x = batch['input_ids']
        else:
            x = batch
        
        # Compute loss
        loss = self.diffusion.compute_loss(self.model, x)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Get tokens from batch
        if isinstance(batch, dict):
            x = batch['input_ids']
        else:
            x = batch
        
        # Compute loss
        loss = self.diffusion.compute_loss(self.model, x)
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine learning rate schedule with warmup
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Train D3PM baseline')
    parser.add_argument('--config', type=str, default='d3pm_small_config.yaml',
                       help='Path to config file')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Maximum training steps')
    parser.add_argument('--num-gpus', type=int, default=2,
                       help='Number of GPUs to use')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size per GPU')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable WandB logging')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Checkpoint directory')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert to namespace for compatibility
    from types import SimpleNamespace
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d
    
    config = dict_to_namespace(config)
    
    # Override config with command-line arguments
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    # Set checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = Path(config.training.checkpoint_dir) if hasattr(config.training, 'checkpoint_dir') else \
                        Path('./outputs/baselines/d3pm_small')
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("D3PM Training Configuration")
    print("="*60)
    print(f"Model: {config.model.hidden_size}d, {config.model.num_layers} layers")
    print(f"Diffusion: {config.diffusion.num_timesteps} steps, {config.diffusion.schedule} schedule")
    print(f"Training: {config.training.max_steps} steps, LR {config.training.learning_rate}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("="*60)
    
    # Create data loaders
    print("\nLoading data...")
    
    # Use wrapped dataset paths
    train_data_path = str(Path(__file__).parent.parent.parent / "experiments" / "generalizability" / "data_cache" / "openwebtext_train")
    val_data_path = str(Path(__file__).parent.parent.parent / "experiments" / "generalizability" / "data_cache" / "openwebtext_val")
    
    train_dataset = TextDataset(train_data_path, max_length=1024)
    val_dataset = TextDataset(val_data_path, max_length=1024)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = D3PMWrapper(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {total_params:,}")
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='d3pm-{step:06d}',
            every_n_train_steps=10000,
            save_top_k=-1,  # Save all checkpoints
            save_last=True
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Setup logger
    if not args.no_wandb:
        logger = WandbLogger(
            project='atat-diffusion-baselines',
            name='d3pm_small_baseline',
            save_dir=str(checkpoint_dir)
        )
    else:
        logger = None
    
    # Create trainer
    trainer = pl.Trainer(
        max_steps=config.training.max_steps,
        accelerator='gpu',
        devices=args.num_gpus,
        strategy='ddp' if args.num_gpus > 1 else 'auto',
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.training.gradient_clip if hasattr(config.training, 'gradient_clip') else 1.0,
        val_check_interval=5000,
        log_every_n_steps=100,
        precision='16-mixed',  # Use mixed precision for efficiency
        enable_progress_bar=True
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Final checkpoint: {checkpoint_dir / 'last.ckpt'}")
    print("="*60)


if __name__ == '__main__':
    main()
