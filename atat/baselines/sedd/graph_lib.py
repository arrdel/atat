"""
Graph library for discrete diffusion processes.
Adapted from: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion

Defines the forward diffusion process for discrete state spaces.
"""

import abc
import torch
import torch.nn.functional as F


def sample_categorical(probs, method="hard"):
    """Sample from categorical distribution."""
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
        return (probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} not supported")


class Graph(abc.ABC):
    """Base class for discrete diffusion graphs."""

    @property
    @abc.abstractmethod
    def dim(self):
        """Dimension of the discrete state space."""
        pass

    @property
    @abc.abstractmethod
    def absorb(self):
        """Whether the graph has an absorbing state."""
        pass

    @abc.abstractmethod
    def rate(self, i):
        """Forward rate matrix Q[i, :]."""
        pass

    @abc.abstractmethod
    def transp_rate(self, i):
        """Transpose rate matrix Q[:, i]."""
        pass

    @abc.abstractmethod
    def transition(self, i, sigma):
        """Transition matrix e^{sigma * Q}[i, :]."""
        pass

    def transp_transition(self, i, sigma):
        """Transpose transition matrix e^{sigma * Q}[:, i]."""
        raise NotImplementedError("Subclass must implement transp_transition")

    def sample_transition(self, i, sigma):
        """Sample from transition distribution."""
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")

    def reverse_rate(self, i, score):
        """Construct reverse rate: score * Q^T."""
        normalized_rate = self.transp_rate(i) * score
        
        # Zero out diagonal, then set diagonal to negative row sum
        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def sample_rate(self, i, rate):
        """Sample from rate distribution."""
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)

    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """Compute staggered score for denoising."""
        pass

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """Sample from limiting distribution."""
        pass

    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """Compute score entropy loss."""
        pass


class Uniform(Graph):
    """
    Uniform graph: everything transitions to everything else uniformly.
    Normalized by dimension to avoid blowup.
    """

    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def absorb(self):
        return False

    def rate(self, i):
        """Rate matrix: uniform transition to all other states."""
        rate = torch.ones_like(i)[..., None].float()
        rate = rate.expand(*i.shape, self.dim) / self.dim
        rate = rate.clone()
        rate.scatter_(-1, i[..., None], torch.zeros_like(rate[..., :1]))
        return rate

    def transp_rate(self, i):
        """Same as rate for uniform graph."""
        return self.rate(i)

    def transition(self, i, sigma):
        """Transition matrix for uniform graph."""
        sigma = sigma[..., None]
        edge_weight = torch.where(
            sigma < 0.5,
            torch.expm1(sigma) / self.dim,
            (torch.exp(sigma) - 1) / self.dim
        )
        
        transition = edge_weight.expand(*i.shape, self.dim).clone()
        transition.scatter_(-1, i[..., None], torch.zeros_like(transition[..., :1]))
        
        diag = 1 - (self.dim - 1) * edge_weight
        transition.scatter_(-1, i[..., None], diag)
        
        return transition

    def transp_transition(self, i, sigma):
        """Same as transition for uniform graph."""
        return self.transition(i, sigma)

    def sample_transition(self, i, sigma):
        """Fast sampling for uniform graph."""
        # Ensure sigma has right shape for broadcasting: (batch,) or (batch, 1) -> (batch, 1)
        if sigma.ndim == 1:
            sigma = sigma[:, None]
        
        move_chance = 1 - torch.exp(-sigma)  # (batch, 1)
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        
        i_pert = torch.where(
            move_indices,
            torch.randint_like(i, self.dim),
            i
        )
        return i_pert

    def staggered_score(self, score, dsigma):
        """Staggered score approximation."""
        return torch.exp(-dsigma[..., None]) * score

    def sample_limit(self, *batch_dims):
        """Sample from uniform distribution."""
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0):
        """Score entropy for uniform graph."""
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)

        # Negative term
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )

        # Constant factor
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim 
        )

        # Positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        
        return pos_term - neg_term + const


class Absorbing(Graph):
    """
    Absorbing graph: all states transition to a single absorbing state (mask token).
    Used for masked language modeling.
    """

    def __init__(self, dim):
        """
        Args:
            dim: vocabulary size + 1 (for absorbing state)
        """
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def absorb(self):
        return True

    def rate(self, i):
        """Rate matrix: all states go to absorbing state."""
        rate = torch.zeros(*i.shape, self.dim, device=i.device)
        rate[..., -1] = 1.0  # Transition to absorbing state
        rate.scatter_(-1, i[..., None], torch.zeros_like(rate[..., :1]))
        return rate

    def transp_rate(self, i):
        """Transpose rate: absorbing state goes nowhere."""
        rate = torch.where(
            i == self.dim - 1,
            torch.zeros(*i.shape, self.dim, device=i.device),
            torch.ones(*i.shape, self.dim, device=i.device)
        )
        rate.scatter_(-1, i[..., None], torch.zeros_like(rate[..., :1]))
        return rate

    def transition(self, i, sigma):
        """Transition matrix for absorbing state."""
        sigma = sigma[..., None]
        edge_weight = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        
        transition = torch.zeros(*i.shape, self.dim, device=i.device)
        transition.scatter_(-1, torch.full_like(i[..., None], self.dim - 1), edge_weight)
        
        # Stay at current state with remaining probability
        diag = 1 - edge_weight
        transition.scatter_(-1, i[..., None], diag)
        
        return transition

    def transp_transition(self, i, sigma):
        """Transpose transition for absorbing state."""
        sigma = sigma[..., None]
        edge_weight = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        
        # Only absorbing state can transition back
        transition = torch.where(
            i[..., None] == self.dim - 1,
            edge_weight.expand(*i.shape, self.dim),
            torch.zeros(*i.shape, self.dim, device=i.device)
        )
        
        # Diagonal: stay at current state
        diag = torch.where(
            i == self.dim - 1,
            torch.ones_like(sigma[..., 0]),
            1 - edge_weight[..., 0]
        )
        transition.scatter_(-1, i[..., None], diag[..., None])
        
        return transition

    def sample_transition(self, i, sigma):
        """Sample transition: move to absorbing state with probability."""
        # Ensure sigma has right shape for broadcasting: (batch,) or (batch, 1) -> (batch, 1)
        if sigma.ndim == 1:
            sigma = sigma[:, None]
        
        move_chance = 1 - torch.exp(-sigma)  # (batch, 1)
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        
        i_pert = torch.where(
            move_indices,
            torch.full_like(i, self.dim - 1),  # Move to absorbing state
            i
        )
        return i_pert

    def staggered_score(self, score, dsigma):
        """Staggered score for absorbing state."""
        return torch.exp(-dsigma[..., None]) * score

    def sample_limit(self, *batch_dims):
        """Limiting distribution is the absorbing state."""
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, sigma, x, x0):
        """Score entropy for absorbing state."""
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        # Negative term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        # Positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

        # Constant term
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy


def get_graph(config, device):
    """Factory function to create graph based on config."""
    graph_type = config.graph.type if hasattr(config, 'graph') else config.get('graph', {}).get('type', 'absorb')
    vocab_size = config.tokens if hasattr(config, 'tokens') else config.get('tokens', 50257)
    
    if graph_type == "uniform":
        return Uniform(vocab_size)
    elif graph_type == "absorb":
        return Absorbing(vocab_size + 1)  # +1 for absorbing state
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
