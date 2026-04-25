"""
Timestep Curriculum Sampler for DDPM Training
==============================================

Implements the Smooth Curriculum strategy described in 正式版本.md:
- Rank coordinate (z) in log-SNR space for uniform timestep density
- Three curriculum directions: Low→High, High→Low, Mid→Ends
- Gaussian kernel soft boundary + uniform floor
- Matched-static control (fixed marginal distribution)

Only changes timestep sampling. Model, loss, optimizer, inference untouched.
"""

import torch
import numpy as np


class CurriculumSampler:
    """
    Timestep sampler supporting curriculum learning and matched-static controls.

    Modes:
        'uniform'      - Standard uniform sampling (baseline)
        'low_to_high'  - Curriculum: low noise → high noise
        'high_to_low'  - Curriculum: high noise → low noise
        'mid_to_ends'  - Curriculum: mid noise → both ends
        'static'       - Fixed distribution (matched-static control)

    Args:
        num_timesteps:   T, total number of diffusion timesteps
        total_train_steps: S, total training steps
        mode:            one of the modes above
        curriculum_ratio: c, fraction of training spent in curriculum phase (default 0.5)
        sigma_soft:      soft boundary width for Gaussian kernel (default 0.08)
        rho:             uniform floor mixing weight (default 0.1)
        beta_start:      beta_1 for linear schedule (default 1e-4)
        beta_end:        beta_T for linear schedule (default 0.02)
        static_marginal: precomputed marginal for 'static' mode (numpy array of shape [T])
    """

    def __init__(
        self,
        num_timesteps: int,
        total_train_steps: int,
        mode: str = 'uniform',
        curriculum_ratio: float = 0.5,
        sigma_soft: float = 0.08,
        rho: float = 0.1,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        static_marginal: np.ndarray = None,
    ):
        assert mode in ('uniform', 'low_to_high', 'high_to_low', 'mid_to_ends', 'static')
        if mode == 'static':
            assert static_marginal is not None, "static mode requires precomputed marginal"

        self.T = num_timesteps
        self.S = total_train_steps
        self.mode = mode
        self.c = curriculum_ratio
        self.sigma_soft = sigma_soft
        self.rho = rho

        # Precompute rank coordinates z_i from linear beta schedule
        self.z = self._compute_rank_coordinates(num_timesteps, beta_start, beta_end)

        # Precompute static marginal if in static mode
        if mode == 'static':
            self._static_probs = torch.from_numpy(
                static_marginal / static_marginal.sum()
            ).float()
        else:
            self._static_probs = None

    @staticmethod
    def _compute_rank_coordinates(T: int, beta_start: float, beta_end: float) -> np.ndarray:
        """
        Compute rank coordinate z_i for each timestep.

        z_i = rank(lambda_i) / (T - 1)

        where lambda_i = log(alpha_bar_i / (1 - alpha_bar_i)) is the log-SNR,
        and rank is the ascending order (smallest lambda = rank 0 = z=0 = highest noise).

        Returns:
            z: numpy array of shape [T], z_i in [0, 1]
        """
        betas = np.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas)
        # Clip to avoid log(0) or log(inf)
        alpha_bar = np.clip(alpha_bar, 1e-10, 1 - 1e-10)
        log_snr = np.log(alpha_bar / (1.0 - alpha_bar))
        # Ascending rank: smallest log_snr (highest noise) gets rank 0, z=0
        ranks = np.argsort(np.argsort(log_snr)).astype(np.float64)
        z = ranks / (T - 1)
        return z

    def _distance(self, z: np.ndarray, uc: float) -> np.ndarray:
        """
        Compute distance D_u(z_i) for the current curriculum direction.

        Args:
            z:  rank coordinates, shape [T]
            uc: curriculum progress u_c in [0, 1]

        Returns:
            D: distance array, shape [T], >= 0
        """
        if self.mode == 'low_to_high':
            # Expand from z=1 (low noise) toward z=0 (high noise)
            # Covered region: [1 - uc, 1]
            return np.maximum(1.0 - uc - z, 0.0)

        elif self.mode == 'high_to_low':
            # Expand from z=0 (high noise) toward z=1 (low noise)
            # Covered region: [0, uc]
            return np.maximum(z - uc, 0.0)

        elif self.mode == 'mid_to_ends':
            # Expand from z=0.5 toward both ends
            # Covered region: [0.5 - 0.5*uc, 0.5 + 0.5*uc]
            return np.maximum(np.abs(z - 0.5) - 0.5 * uc, 0.0)

        else:
            raise ValueError(f"No distance function for mode '{self.mode}'")

    def _curriculum_probs(self, step: int) -> np.ndarray:
        """
        Compute the curriculum sampling probabilities q_s(t_i) for a given step.

        For step < c*S: soft curriculum distribution mixed with uniform floor
        For step >= c*S: pure uniform

        Args:
            step: current training step (0-indexed)

        Returns:
            probs: numpy array of shape [T], sums to 1
        """
        curriculum_steps = self.c * self.S

        if step >= curriculum_steps:
            # Post-curriculum phase: uniform
            return np.ones(self.T) / self.T

        # Curriculum phase
        uc = step / curriculum_steps  # u_c in [0, 1)

        D = self._distance(self.z, uc)

        # Gaussian kernel: exp(-D^2 / (2 * sigma^2))
        G = np.exp(-D**2 / (2.0 * self.sigma_soft**2))
        G_sum = G.sum()
        if G_sum > 0:
            G = G / G_sum
        else:
            G = np.ones(self.T) / self.T

        # Mix with uniform floor
        U = np.ones(self.T) / self.T
        q = (1.0 - self.rho) * G + self.rho * U

        return q

    def sample(self, batch_size: int, step: int, device: torch.device) -> torch.Tensor:
        """
        Sample timesteps for a training batch.

        Args:
            batch_size: number of timesteps to sample
            step:       current training step (0-indexed)
            device:     torch device

        Returns:
            t: LongTensor of shape [batch_size], values in [0, T)
        """
        if self.mode == 'uniform':
            return torch.randint(0, self.T, (batch_size,), device=device).long()

        elif self.mode == 'static':
            return torch.multinomial(
                self._static_probs.to(device), batch_size, replacement=True
            )

        else:
            # Curriculum modes: low_to_high, high_to_low, mid_to_ends
            probs = self._curriculum_probs(step)
            probs_tensor = torch.from_numpy(probs).float().to(device)
            return torch.multinomial(probs_tensor, batch_size, replacement=True)

    def compute_marginal(self, num_steps: int = None) -> np.ndarray:
        """
        Theoretically precompute the cumulative marginal M_i for the entire training.

        M_i = (1/S) * sum_{s=0}^{S-1} q_s(t_i)

        Used to create the matched-static control experiment.

        Args:
            num_steps: number of integration steps.
                       Default None = exact (enumerate all S steps).
                       Set to e.g. 10000 for a fast approximate preview.

        Returns:
            M: numpy array of shape [T], sums to 1
        """
        if self.mode == 'uniform':
            return np.ones(self.T) / self.T

        if self.mode == 'static':
            return self._static_probs.numpy().copy()

        # Default: exact enumeration over all S training steps
        if num_steps is None or num_steps >= self.S:
            M = np.zeros(self.T)
            for s in range(self.S):
                M += self._curriculum_probs(s)
        else:
            # Approximate: sample num_steps evenly spaced points
            M = np.zeros(self.T)
            eval_steps = np.linspace(0, self.S - 1, num_steps).astype(int)
            for s in eval_steps:
                M += self._curriculum_probs(s)

        M /= M.sum()
        return M

    @classmethod
    def create_static_from_curriculum(
        cls,
        num_timesteps: int,
        total_train_steps: int,
        curriculum_mode: str,
        curriculum_ratio: float = 0.5,
        sigma_soft: float = 0.08,
        rho: float = 0.1,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> 'CurriculumSampler':
        """
        Factory method: create a matched-static sampler from a curriculum config.

        First computes the curriculum's theoretical marginal M_i,
        then creates a 'static' sampler that uses M_i as fixed distribution.

        This ensures:
            Curriculum-X vs Static-X have identical cumulative marginals,
            differing only in whether there is temporal ordering.
        """
        # Create a temporary curriculum sampler to compute its marginal
        temp = cls(
            num_timesteps=num_timesteps,
            total_train_steps=total_train_steps,
            mode=curriculum_mode,
            curriculum_ratio=curriculum_ratio,
            sigma_soft=sigma_soft,
            rho=rho,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        marginal = temp.compute_marginal()

        # Create a static sampler using that marginal
        return cls(
            num_timesteps=num_timesteps,
            total_train_steps=total_train_steps,
            mode='static',
            static_marginal=marginal,
            # These don't matter for static mode, but keep for consistency
            curriculum_ratio=curriculum_ratio,
            sigma_soft=sigma_soft,
            rho=rho,
            beta_start=beta_start,
            beta_end=beta_end,
        )
