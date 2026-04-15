"""
Sampling functions for SEDD.
Adapted from: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
"""

import abc
import torch
from .graph_lib import sample_categorical


class Predictor(abc.ABC):
    """Base class for predictor algorithms."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """
        One update of the predictor.

        Args:
            score_fn: score function
            x: current state
            t: current timestep
            step_size: size of time step

        Returns:
            x: next state
        """
        pass


class EulerPredictor(Predictor):
    """Euler predictor for discrete diffusion."""
    
    def update_fn(self, score_fn, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x


class NonePredictor(Predictor):
    """No-op predictor (returns input unchanged)."""
    
    def update_fn(self, score_fn, x, t, step_size):
        return x


class AnalyticPredictor(Predictor):
    """
    Analytic predictor using staggered score.
    More efficient than Euler for discrete diffusion.
    """
    
    def update_fn(self, score_fn, x, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)


class Denoiser:
    """Final denoising step."""
    
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t):
        """
        Denoise to clean sample.
        
        Args:
            score_fn: score function
            x: noisy state
            t: current timestep
        
        Returns:
            x_0: denoised state
        """
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        
        # Truncate probabilities if absorbing state
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        # Return argmax (deterministic denoising)
        return probs.argmax(dim=-1)


def get_predictor(name):
    """Get predictor by name."""
    if name == "euler":
        return EulerPredictor
    elif name == "analytic":
        return AnalyticPredictor
    elif name == "none":
        return NonePredictor
    else:
        raise ValueError(f"Unknown predictor: {name}")


def get_pc_sampler(graph, noise, batch_dims, predictor='analytic', steps=128, 
                   denoise=True, eps=1e-5, device='cuda', proj_fun=None):
    """
    Create a predictor-corrector sampler.
    
    Args:
        graph: discrete diffusion graph
        noise: noise schedule
        batch_dims: tuple of batch dimensions (batch_size, seq_len)
        predictor: predictor type ('euler', 'analytic', 'none')
        steps: number of sampling steps
        denoise: whether to perform final denoising
        eps: minimum timestep
        device: device to sample on
        proj_fun: optional projection function for constrained sampling
    
    Returns:
        sampling_fn: function that samples from the model
    """
    predictor_cls = get_predictor(predictor)
    predictor_obj = predictor_cls(graph, noise)
    denoiser = Denoiser(graph, noise)
    
    def sampling_fn(model):
        """
        Sample from the model.
        
        Args:
            model: SEDD model
        
        Returns:
            samples: (batch_size, seq_len) generated tokens
        """
        from .model import get_score_fn
        
        # Get score function in sampling mode
        score_fn = get_score_fn(model, train=False, sampling=True)
        
        # Initialize from limiting distribution
        x = graph.sample_limit(*batch_dims).to(device)
        
        # Reverse diffusion
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        
        for i in range(steps):
            t = timesteps[i]
            step_size = timesteps[i] - timesteps[i + 1]
            
            # Predictor step
            x = predictor_obj.update_fn(score_fn, x, t * torch.ones(x.shape[0], device=device), step_size)
            
            # Optional projection (for constrained sampling)
            if proj_fun is not None:
                x = proj_fun(x)
        
        # Final denoising
        if denoise:
            t = timesteps[-1]
            x = denoiser.update_fn(score_fn, x, t * torch.ones(x.shape[0], device=device))
            
            if proj_fun is not None:
                x = proj_fun(x)
        
        return x
    
    return sampling_fn


def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    """
    Create sampling function from config.
    
    Args:
        config: configuration
        graph: discrete diffusion graph
        noise: noise schedule
        batch_dims: batch dimensions
        eps: minimum timestep
        device: device
    
    Returns:
        sampling_fn: sampling function
    """
    if hasattr(config, 'sampling'):
        predictor = config.sampling.predictor if hasattr(config.sampling, 'predictor') else 'analytic'
        steps = config.sampling.steps if hasattr(config.sampling, 'steps') else 128
        denoise = config.sampling.noise_removal if hasattr(config.sampling, 'noise_removal') else True
    else:
        predictor = config.get('sampling', {}).get('predictor', 'analytic')
        steps = config.get('sampling', {}).get('steps', 128)
        denoise = config.get('sampling', {}).get('noise_removal', True)
    
    return get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=batch_dims,
        predictor=predictor,
        steps=steps,
        denoise=denoise,
        eps=eps,
        device=device
    )
