"""
Loss functions for SEDD training.
Adapted from: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
"""

import torch
import torch.optim as optim
from .model import get_score_fn


def get_loss_fn(noise, graph, train, sampling_eps=1e-3):
    """
    Create the score entropy loss function.
    
    Args:
        noise: noise schedule
        graph: discrete diffusion graph
        train: whether in training mode
        sampling_eps: minimum timestep for sampling
    
    Returns:
        loss_fn: function that computes loss
    """

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        """
        Compute score entropy loss.
        
        Args:
            model: SEDD model
            batch: (batch_size, seq_len) clean tokens
            cond: optional conditioning
            t: optional timesteps (if None, sample uniformly)
            perturbed_batch: optional pre-perturbed batch
        
        Returns:
            loss: (batch_size,) per-sample losses
        """
        # Sample timesteps uniformly in [sampling_eps, 1]
        if t is None:
            t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
        
        # Get noise at timestep t
        sigma, dsigma = noise(t)
        
        # Perturb the batch: x_t ~ q(x_t | x_0)
        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

        # Get log-score from model
        log_score_fn = get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)
        
        # Compute score entropy
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)

        # Weight by noise rate and sum over sequence
        loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss

    return loss_fn


def get_optimizer(config, params):
    """Create optimizer based on config."""
    if hasattr(config, 'optim'):
        lr = config.optim.lr if hasattr(config.optim, 'lr') else 3e-4
        beta1 = config.optim.beta1 if hasattr(config.optim, 'beta1') else 0.9
        beta2 = config.optim.beta2 if hasattr(config.optim, 'beta2') else 0.999
        weight_decay = config.optim.weight_decay if hasattr(config.optim, 'weight_decay') else 0.01
    else:
        lr = config.get('optim', {}).get('lr', 3e-4)
        beta1 = config.get('optim', {}).get('beta1', 0.9)
        beta2 = config.get('optim', {}).get('beta2', 0.999)
        weight_decay = config.get('optim', {}).get('weight_decay', 0.01)
    
    return optim.AdamW(params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)


def optimization_manager(config):
    """Returns an optimize_fn based on config."""
    
    def optimize_fn(optimizer, scaler, params, step=None):
        """
        Perform optimization step.
        
        Args:
            optimizer: PyTorch optimizer
            scaler: GradScaler for mixed precision
            params: model parameters
            step: current training step (optional)
        """
        # Unscale gradients
        scaler.unscale_(optimizer)
        
        # Clip gradients
        if hasattr(config, 'training'):
            grad_clip = config.training.gradient_clip if hasattr(config.training, 'gradient_clip') else 1.0
        else:
            grad_clip = config.get('training', {}).get('gradient_clip', 1.0)
        
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum=1):
    """
    Create a single training/evaluation step function.
    
    Args:
        noise: noise schedule
        graph: discrete diffusion graph  
        train: whether in training mode
        optimize_fn: optimization function
        accum: number of gradient accumulation steps
    
    Returns:
        step_fn: function that performs one step
    """
    loss_fn = get_loss_fn(noise, graph, train)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        """
        Perform one training or evaluation step.
        
        Args:
            state: dict with 'model', 'optimizer', 'scaler', 'ema', 'step'
            batch: input batch
            cond: optional conditioning
        
        Returns:
            loss: scalar loss value
        """
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            
            # Compute loss (divide by accum for gradient accumulation)
            loss = loss_fn(model, batch, cond=cond).mean() / accum
            
            # Backward pass
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            
            # Update weights after accumulation steps
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            # Evaluation: use EMA weights
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn
