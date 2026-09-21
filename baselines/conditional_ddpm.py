"""Small conditional epsilon-DDPM reference, NOT a reproduction of scDiff.

An independent-cell residual MLP conditions on a GenePT descriptor and the
observed context control mean. There are no paired treated/control trajectories,
population attention, graph layers, reward losses, or test-time-trained weights.
"""

import math

import torch
from torch import nn


def cosine_alphas(steps):
    if steps < 2:
        raise ValueError("At least two training diffusion steps required")
    t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
    curve = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
    curve = curve / curve[0]
    betas = (1 - curve[1:] / curve[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0).float()


class ConditionalDDPM(nn.Module):
    def __init__(self, genes, descriptor_dim, width=512, depth=4, timesteps=1000):
        super().__init__()
        self.config = dict(
            genes=genes,
            descriptor_dim=descriptor_dim,
            width=width,
            depth=depth,
            timesteps=timesteps,
        )
        self.time = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width))
        self.expression = nn.Linear(genes, width)
        self.control = nn.Linear(genes, width)
        self.perturbation = nn.Linear(descriptor_dim, width)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(width),
                    nn.SiLU(),
                    nn.Linear(width, width),
                    nn.SiLU(),
                    nn.Linear(width, width),
                )
                for _ in range(depth)
            ]
        )
        self.output = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, genes))
        self.register_buffer("alphas", cosine_alphas(timesteps))

    def forward(self, x, t, descriptor, control_mean):
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(32, device=x.device, dtype=x.dtype) / 31
        )
        phase = t[:, None].to(x.dtype) * frequencies[None]
        condition = (
            self.time(torch.cat([phase.sin(), phase.cos()], dim=-1))
            + self.perturbation(descriptor)
            + self.control(control_mean)
        )
        h = self.expression(x) + condition
        for block in self.blocks:
            h = (h + block(h + condition)) / math.sqrt(2)
        return self.output(h)


class ConditionalDDPMSampler:
    population_native = False

    def __init__(self, model, *, sampling_steps=100, eta=0.0):
        if not 2 <= sampling_steps <= len(model.alphas):
            raise ValueError("Sampling steps must lie between 2 and training timesteps")
        if not 0 <= eta <= 1:
            raise ValueError("eta must be in [0, 1]")
        self.model = model.eval()
        self.schedule = torch.linspace(0, len(model.alphas) - 1, sampling_steps).round().long()
        self.eta = eta
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    @property
    def num_timesteps(self):
        return len(self.schedule)

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    def denoise_step(self, x_t, t, condition, prev_pred=None):
        times = self.schedule.to(t.device)[t]
        previous = self.schedule.to(t.device)[(t - 1).clamp_min(0)]
        a = self.model.alphas[times, None]
        a_previous = torch.where(t[:, None] > 0, self.model.alphas[previous, None], 1.0)
        epsilon = self.model(x_t, times, condition["descriptor"], condition["control_mean"])
        x0 = (x_t - (1 - a).sqrt() * epsilon) / a.sqrt()
        sigma = self.eta * (((1 - a_previous) / (1 - a)) * (1 - a / a_previous)).clamp_min(0).sqrt()
        x_prev = (
            a_previous.sqrt() * x0 + (1 - a_previous - sigma.square()).clamp_min(0).sqrt() * epsilon
        )
        if self.eta:
            x_prev = x_prev + sigma * torch.randn_like(x_t)
        return {"x_prev": x_prev, "x0_pred": x0}
