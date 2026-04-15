"""
GIF Visualization for ATAT Sampling Process

Creates animated GIFs showing:
1. Progressive unmasking during generation
2. Token-by-token appearance
3. Importance scores overlay
4. Uncertainty heatmap evolution

Similar to the visualization in the MDLM paper.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Optional, Tuple, Dict
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import io


class DiffusionGIFVisualizer:
    """
    Creates animated GIFs of the diffusion sampling process.
    
    Shows tokens being progressively unmasked with optional:
    - Importance scores
    - Uncertainty heatmap
    - Timestep information
    """
    
    def __init__(
        self,
        tokenizer,
        figsize: Tuple[int, int] = (16, 6),
        fps: int = 5,
        font_size: int = 14,
        show_importance: bool = True,
        show_uncertainty: bool = True,
    ):
        """
        Args:
            tokenizer: Tokenizer for decoding tokens
            figsize: Figure size (width, height) in inches
            fps: Frames per second for GIF
            font_size: Base font size for text
            show_importance: Whether to show importance scores
            show_uncertainty: Whether to show uncertainty values
        """
        self.tokenizer = tokenizer
        self.figsize = figsize
        self.fps = fps
        self.font_size = font_size
        self.show_importance = show_importance
        self.show_uncertainty = show_uncertainty
        
        # Color schemes
        self.masked_color = '#CCCCCC'  # Gray for masked tokens
        self.revealed_color = '#4CAF50'  # Green for revealed tokens
        self.importance_cmap = plt.cm.RdYlGn_r  # Red (important) to Green (easy)
        self.uncertainty_cmap = plt.cm.YlOrRd  # Yellow (certain) to Red (uncertain)
        
    def create_sampling_gif(
        self,
        trajectory: List[torch.Tensor],
        importance_trajectory: Optional[List[torch.Tensor]] = None,
        uncertainty_trajectory: Optional[List[torch.Tensor]] = None,
        timesteps: Optional[List[float]] = None,
        save_path: str = "diffusion_sampling.gif",
        max_tokens_display: int = 50,
    ):
        """
        Create GIF showing the full sampling trajectory.
        
        Args:
            trajectory: List of token tensors at each step [(seq_len,), ...]
            importance_trajectory: Optional list of importance scores
            uncertainty_trajectory: Optional list of uncertainty scores
            timesteps: Optional list of timestep values
            save_path: Path to save GIF
            max_tokens_display: Maximum tokens to display (truncate if longer)
        """
        # Prepare data
        n_steps = len(trajectory)
        trajectory_np = [t.cpu().numpy() for t in trajectory]
        
        # Truncate if too long
        if trajectory_np[0].shape[0] > max_tokens_display:
            trajectory_np = [t[:max_tokens_display] for t in trajectory_np]
            if importance_trajectory:
                importance_trajectory = [imp[:max_tokens_display] for imp in importance_trajectory]
            if uncertainty_trajectory:
                uncertainty_trajectory = [unc[:max_tokens_display] for unc in uncertainty_trajectory]
        
        # Decode tokens
        token_strs_trajectory = []
        for tokens in trajectory_np:
            token_strs = []
            for tok in tokens:
                if tok == self.tokenizer.mask_token_id or tok >= self.tokenizer.vocab_size:
                    token_strs.append("[MASK]")
                else:
                    token_strs.append(self.tokenizer.decode([int(tok)]))
            token_strs_trajectory.append(token_strs)
        
        # Create figure
        if self.show_importance or self.show_uncertainty:
            fig = plt.figure(figsize=self.figsize)
            gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
            ax_text = fig.add_subplot(gs[0])
            ax_importance = fig.add_subplot(gs[1]) if self.show_importance else None
            ax_uncertainty = fig.add_subplot(gs[2]) if self.show_uncertainty else None
        else:
            fig, ax_text = plt.subplots(1, 1, figsize=self.figsize)
            ax_importance = None
            ax_uncertainty = None
        
        # Animation function
        def update(frame):
            # Clear axes
            ax_text.clear()
            if ax_importance:
                ax_importance.clear()
            if ax_uncertainty:
                ax_uncertainty.clear()
            
            # Get current state
            tokens = trajectory_np[frame]
            token_strs = token_strs_trajectory[frame]
            is_masked = (tokens == self.tokenizer.mask_token_id) | (tokens >= self.tokenizer.vocab_size)
            
            # Plot tokens
            self._plot_tokens(
                ax_text, token_strs, is_masked,
                title=f"Step {frame}/{n_steps-1}" + 
                      (f" (t={timesteps[frame]:.3f})" if timesteps else "")
            )
            
            # Plot importance
            if ax_importance and importance_trajectory and frame < len(importance_trajectory):
                importance = importance_trajectory[frame].cpu().numpy()
                self._plot_heatmap(
                    ax_importance, importance, token_strs,
                    cmap=self.importance_cmap,
                    title="Token Importance",
                    vmin=0, vmax=1
                )
            
            # Plot uncertainty
            if ax_uncertainty and uncertainty_trajectory and frame < len(uncertainty_trajectory):
                uncertainty = uncertainty_trajectory[frame].cpu().numpy()
                # Normalize uncertainty for visualization
                if uncertainty.max() > 0:
                    uncertainty_norm = uncertainty / uncertainty.max()
                else:
                    uncertainty_norm = uncertainty
                self._plot_heatmap(
                    ax_uncertainty, uncertainty_norm, token_strs,
                    cmap=self.uncertainty_cmap,
                    title="Uncertainty",
                    vmin=0, vmax=1
                )
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=n_steps, interval=1000//self.fps)
        
        # Save as GIF
        writer = PillowWriter(fps=self.fps)
        anim.save(save_path, writer=writer)
        plt.close(fig)
        
        print(f"✓ Saved sampling GIF to: {save_path}")
        return save_path
    
    def _plot_tokens(self, ax, token_strs, is_masked, title=""):
        """Plot tokens with color coding for masked vs revealed."""
        n_tokens = len(token_strs)
        
        # Calculate positions for text wrapping
        max_per_row = 10
        n_rows = (n_tokens + max_per_row - 1) // max_per_row
        
        ax.set_xlim(-0.5, max_per_row - 0.5)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.invert_yaxis()
        
        # Plot each token
        for i, (token_str, masked) in enumerate(zip(token_strs, is_masked)):
            row = i // max_per_row
            col = i % max_per_row
            
            # Color based on masked status
            color = self.masked_color if masked else self.revealed_color
            
            # Draw box
            rect = mpatches.Rectangle(
                (col - 0.4, row - 0.4), 0.8, 0.8,
                linewidth=1, edgecolor='black', facecolor=color, alpha=0.6
            )
            ax.add_patch(rect)
            
            # Draw text
            display_text = token_str if not masked else "?"
            ax.text(
                col, row, display_text,
                ha='center', va='center',
                fontsize=self.font_size,
                weight='bold' if not masked else 'normal',
                color='black' if not masked else '#666666'
            )
        
        ax.set_title(title, fontsize=self.font_size + 2, weight='bold')
        ax.axis('off')
    
    def _plot_heatmap(self, ax, values, token_strs, cmap, title, vmin=0, vmax=1):
        """Plot heatmap of values."""
        n_tokens = len(values)
        
        # Reshape for heatmap (1 row)
        values_2d = values.reshape(1, -1)
        
        # Plot heatmap
        im = ax.imshow(
            values_2d, cmap=cmap, aspect='auto',
            vmin=vmin, vmax=vmax, interpolation='nearest'
        )
        
        # Set ticks
        ax.set_xticks(range(n_tokens))
        ax.set_xticklabels(
            [s[:10] if len(s) > 10 else s for s in token_strs],
            rotation=90, fontsize=self.font_size - 4
        )
        ax.set_yticks([])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, orientation='vertical', pad=0.01, fraction=0.02)
        
        ax.set_title(title, fontsize=self.font_size, weight='bold')


class CompactDiffusionGIF:
    """
    Creates compact, paper-style GIFs similar to MDLM visualization.
    
    Focuses on showing just the text appearing token by token.
    """
    
    def __init__(
        self,
        tokenizer,
        width: int = 800,
        height: int = 200,
        fps: int = 5,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        masked_color: Tuple[int, int, int] = (200, 200, 200),
        highlight_color: Tuple[int, int, int] = (76, 175, 80),  # Green
    ):
        """
        Args:
            tokenizer: Tokenizer for decoding
            width: Image width in pixels
            height: Image height per line
            fps: Frames per second
            bg_color: Background color RGB
            text_color: Text color RGB
            masked_color: Masked token color RGB
            highlight_color: Just-revealed token highlight RGB
        """
        self.tokenizer = tokenizer
        self.width = width
        self.height = height
        self.fps = fps
        self.bg_color = bg_color
        self.text_color = text_color
        self.masked_color = masked_color
        self.highlight_color = highlight_color
        
        # Try to load a nice font
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        except:
            self.font = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
    
    def create_compact_gif(
        self,
        trajectory: List[torch.Tensor],
        save_path: str = "diffusion_compact.gif",
        show_step_numbers: bool = True,
        chars_per_line: int = 80,
    ):
        """
        Create compact GIF showing text generation.
        
        Args:
            trajectory: List of token tensors at each step
            save_path: Path to save GIF
            show_step_numbers: Whether to show step numbers
            chars_per_line: Characters per line for wrapping
        """
        frames = []
        n_steps = len(trajectory)
        
        # Track what changed between frames for highlighting
        prev_tokens = None
        
        for step_idx, tokens in enumerate(trajectory):
            tokens_np = tokens.cpu().numpy()
            
            # Decode tokens
            text_parts = []
            changed_indices = set()
            
            for i, tok in enumerate(tokens_np):
                if tok == self.tokenizer.mask_token_id or tok >= self.tokenizer.vocab_size:
                    text_parts.append("[?]")
                else:
                    decoded = self.tokenizer.decode([int(tok)])
                    text_parts.append(decoded)
                    
                    # Check if this token just changed
                    if prev_tokens is not None:
                        prev_tok = prev_tokens[i]
                        if prev_tok != tok:
                            changed_indices.add(i)
            
            prev_tokens = tokens_np.copy()
            
            # Create image for this frame
            frame = self._create_text_frame(
                text_parts, changed_indices, step_idx, n_steps,
                show_step_numbers, chars_per_line
            )
            frames.append(frame)
        
        # Save as GIF
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // self.fps,
            loop=0
        )
        
        print(f"✓ Saved compact GIF to: {save_path}")
        return save_path
    
    def _create_text_frame(
        self, text_parts, changed_indices, step_idx, n_steps,
        show_step_numbers, chars_per_line
    ):
        """Create a single frame image."""
        # Calculate text layout
        full_text = " ".join(text_parts)
        lines = self._wrap_text(full_text, chars_per_line)
        
        # Calculate image height based on number of lines
        line_height = 30
        margin = 20
        step_height = 30 if show_step_numbers else 0
        img_height = step_height + len(lines) * line_height + margin * 2
        
        # Create image
        img = Image.new('RGB', (self.width, img_height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw step number
        if show_step_numbers:
            step_text = f"Step {step_idx + 1}/{n_steps}"
            draw.text((margin, margin), step_text, fill=self.text_color, font=self.font_small)
        
        # Draw text with highlighting
        y_pos = margin + step_height
        char_idx = 0
        
        for line in lines:
            x_pos = margin
            
            for char in line:
                # Determine color
                if char == '[' or char == '?' or char == ']':
                    color = self.masked_color
                elif char_idx in changed_indices:
                    color = self.highlight_color
                else:
                    color = self.text_color
                
                draw.text((x_pos, y_pos), char, fill=color, font=self.font)
                
                # Rough character width estimation
                x_pos += 12
                if char != ' ':
                    char_idx += 1
            
            y_pos += line_height
        
        return img
    
    def _wrap_text(self, text, chars_per_line):
        """Wrap text into lines."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > chars_per_line:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_length = word_length
                else:
                    # Single word longer than line
                    lines.append(word)
                    current_length = 0
            else:
                current_line.append(word)
                current_length += word_length
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines


def create_side_by_side_comparison_gif(
    baseline_trajectory: List[torch.Tensor],
    atat_trajectory: List[torch.Tensor],
    tokenizer,
    save_path: str = "comparison.gif",
    fps: int = 5,
):
    """
    Create side-by-side comparison of baseline vs ATAT sampling.
    
    Args:
        baseline_trajectory: Baseline sampling trajectory
        atat_trajectory: ATAT sampling trajectory
        tokenizer: Tokenizer
        save_path: Path to save GIF
        fps: Frames per second
    """
    visualizer = CompactDiffusionGIF(tokenizer, width=1600, height=400, fps=fps)
    
    # Make sure trajectories are same length (pad if needed)
    max_len = max(len(baseline_trajectory), len(atat_trajectory))
    
    while len(baseline_trajectory) < max_len:
        baseline_trajectory.append(baseline_trajectory[-1])
    while len(atat_trajectory) < max_len:
        atat_trajectory.append(atat_trajectory[-1])
    
    frames = []
    
    for step in range(max_len):
        # Create frame for baseline
        baseline_img = visualizer._create_text_frame(
            *visualizer._prepare_text_parts(baseline_trajectory[step]),
            step, max_len, True, 40
        )
        
        # Create frame for ATAT
        atat_img = visualizer._prepare_text_parts(atat_trajectory[step])
        atat_img = visualizer._create_text_frame(
            *atat_img, step, max_len, True, 40
        )
        
        # Combine side by side
        combined = Image.new('RGB', (baseline_img.width + atat_img.width, max(baseline_img.height, atat_img.height)), (255, 255, 255))
        combined.paste(baseline_img, (0, 0))
        combined.paste(atat_img, (baseline_img.width, 0))
        
        # Add labels
        draw = ImageDraw.Draw(combined)
        draw.text((10, 10), "Baseline", fill=(0, 0, 0))
        draw.text((baseline_img.width + 10, 10), "ATAT", fill=(0, 0, 0))
        
        frames.append(combined)
    
    # Save
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0
    )
    
    print(f"✓ Saved comparison GIF to: {save_path}")
    return save_path


# Example usage function
def create_sample_visualization(
    model,
    tokenizer,
    prompt: str = "",
    length: int = 50,
    num_steps: int = 50,
    save_path: str = "sampling_visualization.gif",
    style: str = "detailed",  # "detailed" or "compact"
):
    """
    Create visualization GIF from a model sampling process.
    
    Args:
        model: ATAT model
        tokenizer: Tokenizer
        prompt: Optional prompt text
        length: Sequence length to generate
        num_steps: Number of denoising steps
        save_path: Where to save GIF
        style: "detailed" (with importance/uncertainty) or "compact" (text only)
    """
    model.eval()
    
    # Storage for trajectory
    trajectory = []
    importance_trajectory = []
    uncertainty_trajectory = []
    timesteps = []
    
    # Initialize
    device = next(model.parameters()).device
    mask_index = tokenizer.mask_token_id
    
    if prompt:
        # Encode prompt
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        x_t = torch.full((1, length), mask_index, dtype=torch.long, device=device)
        x_t[0, :len(prompt_ids[0])] = prompt_ids[0]
    else:
        x_t = torch.full((1, length), mask_index, dtype=torch.long, device=device)
    
    # Sampling loop
    with torch.no_grad():
        for step in range(num_steps):
            t = torch.tensor([1.0 - step / num_steps], device=device)
            timesteps.append(t.item())
            
            # Store current state
            trajectory.append(x_t[0].clone())
            
            # Get predictions
            logits, importance = model(x_t, t, return_importance=True)
            
            # Compute uncertainty
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            
            # Store
            if importance is not None:
                importance_trajectory.append(importance[0].clone())
            uncertainty_trajectory.append(entropy[0].clone())
            
            # Denoise step (simple version)
            is_masked = (x_t[0] == mask_index)
            if is_masked.any():
                # Select tokens to denoise based on uncertainty
                masked_positions = torch.where(is_masked)[0]
                if len(masked_positions) > 0:
                    k = max(1, len(masked_positions) // (num_steps - step + 1))
                    uncertainties = entropy[0][masked_positions]
                    top_k_indices = torch.topk(uncertainties, min(k, len(masked_positions)))[1]
                    positions_to_denoise = masked_positions[top_k_indices]
                    
                    # Sample tokens
                    for pos in positions_to_denoise:
                        probs_pos = torch.softmax(logits[0, pos], dim=-1)
                        sampled_token = torch.multinomial(probs_pos, 1)
                        x_t[0, pos] = sampled_token
    
    # Final state
    trajectory.append(x_t[0].clone())
    
    # Create visualization
    if style == "detailed":
        visualizer = DiffusionGIFVisualizer(
            tokenizer,
            show_importance=len(importance_trajectory) > 0,
            show_uncertainty=True
        )
        visualizer.create_sampling_gif(
            trajectory=trajectory,
            importance_trajectory=importance_trajectory if importance_trajectory else None,
            uncertainty_trajectory=uncertainty_trajectory,
            timesteps=timesteps,
            save_path=save_path
        )
    else:  # compact
        visualizer = CompactDiffusionGIF(tokenizer)
        visualizer.create_compact_gif(
            trajectory=trajectory,
            save_path=save_path
        )
    
    return save_path
