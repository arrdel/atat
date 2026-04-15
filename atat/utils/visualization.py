"""
Visualization and Analysis Utilities for ATAT

Functions for visualizing importance scores, uncertainty, and analyzing ATAT behavior.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import wandb


def visualize_importance_map(
    tokens: torch.Tensor,
    importance: torch.Tensor,
    tokenizer,
    save_path: Optional[str] = None,
    title: str = "Token Importance Heatmap",
) -> plt.Figure:
    """
    Visualize importance scores as a heatmap.
    
    Args:
        tokens: Token IDs of shape (seq_length,)
        importance: Importance scores of shape (seq_length,)
        tokenizer: Tokenizer for decoding tokens
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy
    tokens = tokens.cpu().numpy()
    importance = importance.cpu().numpy()
    
    # Decode tokens
    token_strs = [tokenizer.decode([t]) for t in tokens]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 3))
    
    # Plot heatmap
    im = ax.imshow(
        importance.reshape(1, -1),
        cmap='RdYlGn',
        aspect='auto',
        vmin=0,
        vmax=1,
    )
    
    # Set ticks
    ax.set_yticks([])
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(token_strs, rotation=90, fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Importance Score', rotation=270, labelpad=15)
    
    # Title
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def visualize_curriculum_progress(
    step_history: List[int],
    easy_frac_history: List[float],
    medium_frac_history: List[float],
    hard_frac_history: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize curriculum learning progress over training.
    
    Args:
        step_history: Training steps
        easy/medium/hard_frac_history: Fraction of each difficulty level
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(step_history, easy_frac_history, label='Easy', color='green', linewidth=2)
    ax.plot(step_history, medium_frac_history, label='Medium', color='orange', linewidth=2)
    ax.plot(step_history, hard_frac_history, label='Hard', color='red', linewidth=2)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Fraction of Tokens')
    ax.set_title('Curriculum Learning Progress')
    ax.legend()
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def visualize_uncertainty_distribution(
    uncertainty: torch.Tensor,
    bins: int = 50,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize distribution of uncertainty scores.
    
    Args:
        uncertainty: Uncertainty scores
        bins: Number of histogram bins
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    uncertainty_np = uncertainty.cpu().numpy().flatten()
    
    ax.hist(uncertainty_np, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(
        uncertainty_np.mean(),
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {uncertainty_np.mean():.3f}',
    )
    
    ax.set_xlabel('Uncertainty Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Uncertainty Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def analyze_importance_correlation(
    importance: torch.Tensor,
    reconstruction_error: torch.Tensor,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Analyze correlation between importance and reconstruction error.
    
    Higher correlation means importance estimator is learning well.
    
    Args:
        importance: Predicted importance scores
        reconstruction_error: Actual reconstruction error
        save_path: Optional path to save figure
        
    Returns:
        Dictionary with correlation metrics
    """
    # Convert to numpy
    importance = importance.cpu().numpy().flatten()
    error = reconstruction_error.cpu().numpy().flatten()
    
    # Compute correlation
    correlation = np.corrcoef(importance, error)[0, 1]
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(importance, error, alpha=0.5, s=10)
    ax.set_xlabel('Predicted Importance')
    ax.set_ylabel('Reconstruction Error')
    ax.set_title(f'Importance vs Error (Correlation: {correlation:.3f})')
    ax.grid(alpha=0.3)
    
    # Add regression line
    z = np.polyfit(importance, error, 1)
    p = np.poly1d(z)
    ax.plot(
        importance,
        p(importance),
        "r--",
        linewidth=2,
        label=f'y={z[0]:.2f}x+{z[1]:.2f}',
    )
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.close(fig)
    
    return {
        'correlation': correlation,
        'slope': z[0],
        'intercept': z[1],
    }


def log_atat_metrics_to_wandb(
    metrics: Dict[str, float],
    importance: Optional[torch.Tensor] = None,
    uncertainty: Optional[torch.Tensor] = None,
    step: int = 0,
):
    """
    Log ATAT-specific metrics to Weights & Biases.
    
    Args:
        metrics: Dictionary of scalar metrics
        importance: Optional importance scores for histogram
        uncertainty: Optional uncertainty scores for histogram
        step: Training step
    """
    # Log scalar metrics
    wandb.log(metrics, step=step)
    
    # Log importance histogram
    if importance is not None:
        wandb.log({
            'importance/histogram': wandb.Histogram(importance.cpu().numpy()),
        }, step=step)
        
    # Log uncertainty histogram
    if uncertainty is not None:
        wandb.log({
            'uncertainty/histogram': wandb.Histogram(uncertainty.cpu().numpy()),
        }, step=step)


def create_comparison_table(
    baseline_metrics: Dict[str, float],
    atat_metrics: Dict[str, float],
) -> str:
    """
    Create a markdown table comparing baseline vs ATAT metrics.
    
    Args:
        baseline_metrics: Metrics from baseline model
        atat_metrics: Metrics from ATAT model
        
    Returns:
        Markdown formatted table
    """
    table = "| Metric | Baseline | ATAT | Improvement |\n"
    table += "|--------|----------|------|-------------|\n"
    
    for key in baseline_metrics:
        if key in atat_metrics:
            baseline_val = baseline_metrics[key]
            atat_val = atat_metrics[key]
            
            # Compute improvement (negative for loss-like metrics)
            if 'loss' in key.lower() or 'perplexity' in key.lower():
                improvement = (baseline_val - atat_val) / baseline_val * 100
                sign = '-' if improvement < 0 else '+'
            else:
                improvement = (atat_val - baseline_val) / baseline_val * 100
                sign = '+' if improvement > 0 else ''
                
            table += f"| {key} | {baseline_val:.4f} | {atat_val:.4f} | {sign}{improvement:.2f}% |\n"
            
    return table


def visualize_sampling_trajectory(
    trajectories: List[torch.Tensor],
    tokenizer,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize the token denoising trajectory during sampling.
    
    Args:
        trajectories: List of token sequences at different timesteps
        tokenizer: Tokenizer for decoding
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_steps = len(trajectories)
    seq_length = trajectories[0].shape[0]
    
    # Create matrix of token IDs over time
    trajectory_matrix = torch.stack(trajectories).cpu().numpy()  # (n_steps, seq_length)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot heatmap
    im = ax.imshow(
        trajectory_matrix,
        cmap='tab20',
        aspect='auto',
        interpolation='nearest',
    )
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Sampling Step')
    ax.set_title('Sampling Trajectory (Token Evolution)')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Token ID')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig
