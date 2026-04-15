"""
Evaluation script for Autoregressive Transformer baseline.

Usage:
    python evaluate_ar.py --checkpoint path/to/checkpoint.ckpt --dataset openwebtext
    python evaluate_ar.py --checkpoint path/to/checkpoint.ckpt --dataset wikitext103
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import argparse
from tqdm import tqdm
import json

from atat.baselines.ar_transformer.model import ARTransformer
from atat.baselines.ar_transformer.train_ar import ARTransformerLightning
from mdlm.dataloader import get_dataloaders


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset.
    
    Returns:
        metrics: Dict with loss, perplexity, bpd
    """
    model.eval()
    model = model.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        
        # Compute loss
        loss, _ = model.compute_loss(input_ids)
        
        # Accumulate
        batch_tokens = input_ids.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
    
    # Compute metrics
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    bpd = avg_loss / torch.log(torch.tensor(2.0))  # bits per dimension
    
    metrics = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "bpd": bpd.item(),
        "num_tokens": total_tokens,
    }
    
    return metrics


@torch.no_grad()
def generate_samples(model, tokenizer, num_samples=10, max_length=100, temperature=0.8):
    """Generate text samples from the model."""
    model.eval()
    
    samples = []
    for i in range(num_samples):
        # Start with a random prompt or BOS token
        prompt = torch.tensor([[tokenizer.bos_token_id]]).to(model.device)
        
        # Generate
        generated = model.generate(
            prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
        )
        
        # Decode
        text = tokenizer.decode(generated[0].cpu().tolist(), skip_special_tokens=True)
        samples.append(text)
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate AR Transformer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--dataset", type=str, default="openwebtext", 
                       choices=["openwebtext", "wikitext103", "ptb", "lm1b"],
                       help="Dataset to evaluate on")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--generate", action="store_true", help="Generate text samples")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output file")
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model = ARTransformerLightning.load_from_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.model.to(device)  # Extract the AR model from Lightning wrapper
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    _, val_loader = get_dataloaders(
        dataset=args.dataset,
        cache_dir="./data_cache",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
    )
    
    # Evaluate
    print("Evaluating...")
    metrics = evaluate_model(model, val_loader, device)
    
    print("\n" + "="*60)
    print(f"Evaluation Results on {args.dataset}")
    print("="*60)
    print(f"Loss:        {metrics['loss']:.4f}")
    print(f"Perplexity:  {metrics['perplexity']:.4f}")
    print(f"BPD:         {metrics['bpd']:.4f}")
    print(f"Num Tokens:  {metrics['num_tokens']:,}")
    print("="*60)
    
    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "metrics": metrics,
    }
    
    # Generate samples if requested
    if args.generate:
        print(f"\nGenerating {args.num_samples} samples...")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            samples = generate_samples(
                model, 
                tokenizer, 
                num_samples=args.num_samples,
                max_length=100,
                temperature=0.8
            )
            
            print("\n" + "="*60)
            print("Generated Samples")
            print("="*60)
            for i, sample in enumerate(samples, 1):
                print(f"\nSample {i}:")
                print(sample)
                print("-"*60)
            
            results["samples"] = samples
        except Exception as e:
            print(f"Error generating samples: {e}")
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
