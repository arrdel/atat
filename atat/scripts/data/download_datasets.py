#!/usr/bin/env python3
"""
ATAT Dataset Downloader

Downloads and caches all required datasets for ATAT training.
This should be run ONCE before any training begins.

Usage:
    python scripts/download_datasets.py [OPTIONS]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from datasets import load_dataset

# Color codes for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

# Dataset configurations: (name, config, description, size_estimate)
CORE_DATASETS = [
    ("Skylion007/openwebtext", "", "Primary training dataset", "~40GB"),
    ("wikitext", "wikitext-103-v1", "Validation dataset", "~500MB"),
    ("wikitext", "wikitext-2-v1", "Quick eval dataset", "~12MB"),
    ("ptb_text_only", "", "Evaluation dataset", "~6MB"),
]

OPTIONAL_DATASETS = [
    ("lm1b", "", "Large-scale evaluation", "~30GB"),
    ("ag_news", "", "Classification task", "~30MB"),
]


def print_header(message: str) -> None:
    """Print a formatted header message."""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}{message.center(60)}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")


def download_dataset(
    dataset_name: str,
    dataset_config: str,
    cache_dir: str,
    description: str = "",
    size_estimate: str = ""
) -> bool:
    """
    Download and cache a single dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration/subset name
        cache_dir: Directory to cache the dataset
        description: Human-readable description
        size_estimate: Estimated download size
    
    Returns:
        True if successful, False otherwise
    """
    desc_str = f"{description} ({size_estimate})" if description and size_estimate else ""
    print(f"\n{Colors.YELLOW}Downloading {dataset_name}{Colors.NC} {desc_str}")
    print(f"Cache directory: {cache_dir}")
    
    try:
        if dataset_config:
            dataset = load_dataset(
                dataset_name,
                dataset_config,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        else:
            dataset = load_dataset(
                dataset_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        
        print(f"{Colors.GREEN}✓ Successfully downloaded{Colors.NC}")
        
        # Show dataset info
        for split in dataset.keys():
            num_examples = len(dataset[split])
            print(f"  - {split}: {num_examples:,} examples")
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}✗ Error downloading {dataset_name}: {e}{Colors.NC}", file=sys.stderr)
        return False


def get_cache_size(cache_dir: str) -> str:
    """Get the total size of the cache directory."""
    import subprocess
    try:
        result = subprocess.run(
            ["du", "-sh", cache_dir],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.split()[0]
    except Exception:
        return "Unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Download and cache datasets for ATAT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./data_cache",
        help="Directory to cache datasets"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for logs"
    )
    
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Download optional evaluation datasets"
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of specific datasets (overrides defaults)"
    )
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print_header("ATAT Dataset Downloader")
    print(f"Data cache: {Colors.GREEN}{args.cache_dir}{Colors.NC}")
    print(f"Logs: {Colors.GREEN}{args.log_dir}{Colors.NC}")
    
    # Download datasets
    if args.datasets:
        # Custom dataset list
        dataset_names = [d.strip() for d in args.datasets.split(",")]
        print(f"\n{Colors.BLUE}Downloading custom dataset list: {dataset_names}{Colors.NC}")
        # Simple download without configs for custom lists
        for dataset_name in dataset_names:
            download_dataset(dataset_name, "", args.cache_dir)
    else:
        # Core datasets
        print(f"\n{Colors.BLUE}=== Core Datasets (Required) ==={Colors.NC}")
        success_count = 0
        for i, (name, config, desc, size) in enumerate(CORE_DATASETS, 1):
            print(f"\n{Colors.YELLOW}[{i}/{len(CORE_DATASETS)}]{Colors.NC} {name}")
            if download_dataset(name, config, args.cache_dir, desc, size):
                success_count += 1
        
        print(f"\n{Colors.GREEN}Downloaded {success_count}/{len(CORE_DATASETS)} core datasets{Colors.NC}")
        
        # Optional datasets
        if args.include_optional:
            print(f"\n{Colors.BLUE}=== Optional Datasets ==={Colors.NC}")
            for i, (name, config, desc, size) in enumerate(OPTIONAL_DATASETS, 1):
                print(f"\n{Colors.YELLOW}[{i}/{len(OPTIONAL_DATASETS)}]{Colors.NC} {name}")
                download_dataset(name, config, args.cache_dir, desc, size)
    
    # Show final cache size
    print_header("Download Complete")
    final_size = get_cache_size(args.cache_dir)
    print(f"Final cache size: {Colors.GREEN}{final_size}{Colors.NC}")
    print(f"Cache location: {Colors.GREEN}{args.cache_dir}{Colors.NC}\n")
    
    print(f"{Colors.BLUE}Next steps:{Colors.NC}")
    print(f"  1. Start training: {Colors.YELLOW}python atat/scripts/training/train.py --config atat/configs/atat/train.yaml{Colors.NC}")
    print(f"  2. Monitor with WandB: {Colors.YELLOW}https://wandb.ai{Colors.NC}")
    print(f"  3. Check logs in: {Colors.YELLOW}{args.log_dir}{Colors.NC}\n")


if __name__ == "__main__":
    main()
