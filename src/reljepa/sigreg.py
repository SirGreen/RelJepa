"""SIGReg (Sketched Isotropic Gaussian Regularization) for the RelJEPA-SIGReg ablation.

Re-implementation following LeJEPA (Balestriero & LeCun, arXiv:2511.08544) and the authors'
MINIMAL.md example (github.com/rbalestr-lab/lejepa, CC BY-NC 4.0)
. Embeddings are projected on random unit directions and the empirical
characteristic function of every 1-D projection is compared with that of a standard Gaussian on
t in [0, 3] (Epps-Pulley-type statistic, trapezoidal quadrature with a Gaussian window, using the
symmetry of the characteristic function). LeJEPA mixes it as lambda * SIGReg + (1 - lambda) * pred.
"""
import torch
from torch import nn


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, t_max: float = 3.0, num_slices: int = 256):
        super().__init__()
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.num_slices = num_slices
        self.register_buffer("t", t)
        self.register_buffer("phi", window)                 # characteristic function of N(0, 1)
        self.register_buffer("weights", weights * window)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [num_samples, dim] -> scalar statistic (mean over random slices)."""
        a = torch.randn(z.size(-1), self.num_slices, device=z.device, dtype=z.dtype)
        a = a / a.norm(p=2, dim=0, keepdim=True)
        x_t = (z @ a).unsqueeze(-1) * self.t                 # [N, slices, knots]
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        statistic = (err @ self.weights) * z.size(0)
        return statistic.mean()
