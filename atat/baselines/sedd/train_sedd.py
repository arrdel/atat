"""
Training script for SEDD baseline.
Integrates SEDD with PyTorch Lightning and ATAT's data pipeline.
"""

import sys
from pathlib import Path

# Add parent directories to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "atat"))

import argparse
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk

# Import SEDD components using absolute imports
from baselines.sedd import model as sedd_model
from baselines.sedd import graph_lib
from baselines.sedd import noise_lib
from baselines.sedd import losses
from baselines.sedd import sampling


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

# EMA implementation
from torch.optim.swa_utils import AveragedModel


class ExponentialMovingAverage:
    """Simple EMA implementation."""
    
    def __init__(self, parameters, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in parameters:
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, parameters):
        for name, param in parameters:
            if param.requires_grad:
                assert name in self.shadow
                # Ensure shadow is on same device as parameter
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def store(self, parameters):
        for name, param in parameters:
            if param.requires_grad:
                self.backup[name] = param.data.clone()
    
    def restore(self, parameters):
        for name, param in parameters:
            if param.requires_grad:
                param.data = self.backup[name]
    
    def copy_to(self, parameters):
        for name, param in parameters:
            if param.requires_grad:
                # Ensure shadow tensor is on same device as parameter
                shadow = self.shadow[name]
                if shadow.device != param.device:
                    shadow = shadow.to(param.device)
                param.data = shadow


class SEDDWrapper(L.LightningModule):
    """PyTorch Lightning wrapper for SEDD."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Create SEDD model
        self.sedd = sedd_model.SEDD(config)
        
        # Create graph and noise (will be properly initialized in setup())
        self.graph = None
        self.noise = noise_lib.get_noise(config)
        
        # Loss function will be created in setup()
        self.loss_fn = None
        
        # EMA
        self.ema = ExponentialMovingAverage(
            self.sedd.named_parameters(),
            decay=config.training.ema if hasattr(config, 'training') else 0.9999
        )
        
        # Scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.train_step_count = 0
    
    def setup(self, stage=None):
        """Called when the model is moved to device."""
        # Now we can create device-dependent components
        if self.graph is None:
            self.graph = graph_lib.get_graph(self.config, self.device)
        if self.loss_fn is None:
            self.loss_fn = losses.get_loss_fn(
                self.noise,
                self.graph,
                train=True,
                sampling_eps=1e-5
            )
    
    def forward(self, indices, sigma):
        return self.sedd(indices, sigma)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Get input_ids from batch and ensure it's on the correct device
        input_ids = batch['input_ids']
        # PyTorch Lightning should handle device placement, but ensure it explicitly
        if not input_ids.is_cuda:
            input_ids = input_ids.to(self.device)
        
        # Compute loss
        loss = self.loss_fn(self.sedd, input_ids).mean()
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train/step', float(self.train_step_count), on_step=True, on_epoch=False)
        
        self.train_step_count += 1
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids = batch['input_ids']
        # Ensure input is on correct device
        if not input_ids.is_cuda:
            input_ids = input_ids.to(self.device)
        
        # Use EMA weights
        self.ema.store(self.sedd.named_parameters())
        self.ema.copy_to(self.sedd.named_parameters())
        
        # Compute loss
        with torch.no_grad():
            val_loss_fn = losses.get_loss_fn(
                self.noise,
                self.graph,
                train=False,
                sampling_eps=1e-5
            )
            loss = val_loss_fn(self.sedd, input_ids).mean()
        
        # Restore training weights
        self.ema.restore(self.sedd.named_parameters())
        
        # Log metrics
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Optimizer
        optimizer = losses.get_optimizer(self.config, self.sedd.parameters())
        
        # Learning rate scheduler
        if hasattr(self.config, 'lr_scheduler'):
            warmup_steps = self.config.lr_scheduler.warmup_steps if hasattr(self.config.lr_scheduler, 'warmup_steps') else 10000
        else:
            warmup_steps = self.config.get('lr_scheduler', {}).get('warmup_steps', 10000)
        
        max_steps = self.config.trainer.max_steps if hasattr(self.config, 'trainer') else 500000
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA after each training batch."""
        self.ema.update(self.sedd.named_parameters())


def main():
    parser = argparse.ArgumentParser(description="Train SEDD baseline")
    parser.add_argument("--config", type=str, default="sedd_baseline_config.yaml",
                       help="Config file path")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Override max training steps")
    parser.add_argument("--num-gpus", type=int, default=2,
                       help="Number of GPUs to use")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable WandB logging")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    # Override config with command line args
    if args.max_steps:
        if not hasattr(config, 'trainer'):
            config.trainer = {}
        config.trainer.max_steps = args.max_steps
    
    if args.no_wandb:
        if not hasattr(config, 'wandb'):
            config.wandb = {}
        config.wandb.mode = "disabled"
    
    print("="*80)
    print("SEDD Baseline Training - Score Entropy Discrete Diffusion")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Max steps: {config.trainer.max_steps}")
    print(f"Batch size per GPU: {config.loader.batch_size}")
    print(f"Global batch size: {config.loader.global_batch_size}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"Graph type: {config.graph.type}")
    print(f"Noise type: {config.noise.type}")
    print("="*80)
    print()
    
    # Get dataloaders
    print("Loading data...")
    train_data_path = str(Path(__file__).parent.parent.parent / "experiments" / "generalizability" / "data_cache" / "openwebtext_train")
    val_data_path = str(Path(__file__).parent.parent.parent / "experiments" / "generalizability" / "data_cache" / "openwebtext_val")
    
    train_dataset = TextDataset(train_data_path, max_length=1024)
    val_dataset = TextDataset(val_data_path, max_length=1024)
    
    # Get batch size from config
    batch_size = config.loader.batch_size if hasattr(config, 'loader') else 4
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
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
    
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    print()
    
    # Create model
    model = SEDDWrapper(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.sedd.parameters())
    trainable_params = sum(p.numel() for p in model.sedd.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Setup callbacks
    callbacks = []
    
    # Checkpointing
    checkpoint_dir = config.checkpointing.save_dir if hasattr(config, 'checkpointing') else "./checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='sedd-{step:06d}',
        every_n_train_steps=config.checkpointing.every_n_train_steps if hasattr(config, 'checkpointing') else 10000,
        save_top_k=config.checkpointing.save_top_k if hasattr(config, 'checkpointing') else 3,
        monitor='val/loss',
        mode='min',
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Setup logger
    logger = None
    if config.wandb.mode != "disabled":
        logger = WandbLogger(
            project=config.wandb.project,
            name=config.wandb.name,
            save_dir=checkpoint_dir,
            log_model=False,
        )
    
    # Create trainer
    # SEDD has unused parameters in some forward passes (e.g. conditional vs unconditional),
    # so we need find_unused_parameters=True for DDP.
    strategy = "auto"
    if args.num_gpus > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = L.Trainer(
        accelerator='gpu',
        devices=args.num_gpus,
        strategy=strategy,
        max_steps=config.trainer.max_steps,
        precision='16-mixed',
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        gradient_clip_val=config.trainer.gradient_clip_val if hasattr(config.trainer, 'gradient_clip_val') else 1.0,
        log_every_n_steps=config.trainer.log_every_n_steps,
        val_check_interval=config.trainer.val_check_interval,
        limit_val_batches=config.trainer.limit_val_batches,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train
    print("Starting training...")
    print()
    trainer.fit(model, train_loader, val_loader)
    
    print("\n✓ Training completed successfully!")


if __name__ == "__main__":
    main()
