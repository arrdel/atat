#!/usr/bin/env python3
"""
Train MDLM Baseline with Uniform Masking

This uses the existing MDLM codebase with uniform random masking (no importance weighting).
This serves as the baseline to compare against ATAT's adaptive masking.

Usage:
    python train_mdlm_baseline.py --max-steps 100  # Debug run
    python train_mdlm_baseline.py  # Full training (500K steps)
"""

import sys
from pathlib import Path

# Add MDLM to path
MDLM_ROOT = Path(__file__).parent.parent.parent.parent / "mdlm"
sys.path.insert(0, str(MDLM_ROOT.parent))

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Train MDLM baseline with uniform masking")
    parser.add_argument("--max-steps", type=int, default=500000,
                       help="Max training steps (default: 500K)")
    parser.add_argument("--num-gpus", type=int, default=2,
                       help="Number of GPUs to use")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable WandB logging")
    args = parser.parse_args()
    
    baseline_dir = Path(__file__).parent
    config_name = "mdlm_baseline_config"
    
    print("="*80)
    print("MDLM Baseline Training - Uniform Random Masking")
    print("="*80)
    print(f"Config: {config_name}.yaml")
    print(f"Max steps: {args.max_steps}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"Masking: UNIFORM (no importance weighting)")
    print(f"Curriculum: NONE (standard MDLM)")
    print("="*80)
    print()
    
    # Build command to run MDLM main.py with our config
    cmd = [
        "python", "-m", "mdlm.main",
        f"--config-path={baseline_dir.absolute()}",
        f"--config-name={config_name}",
        f"trainer.max_steps={args.max_steps}",
        f"trainer.devices={args.num_gpus}",
    ]
    
    if args.no_wandb:
        cmd.append("wandb.mode=disabled")
    
    print(f"Running command:")
    print(f"  cd {MDLM_ROOT.parent}")
    print(f"  {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, cwd=MDLM_ROOT.parent, check=True)
        print("\n✓ Training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n✗ Training interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
