"""Squidiff's native DDIM kernel exposed to the existing population SMC engine.

No modification of the denoiser, semantic condition, or upstream time mapping.
The same adapter is used for vanilla, equal-budget selection and AdaCell runs.
"""

import torch


class SquidiffSampler:
    population_native = False

    def __init__(self, model, diffusion, *, eta=0.0):
        self.model, self.diffusion = model.eval(), diffusion
        self.eta = eta
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    @property
    def num_timesteps(self):
        return self.diffusion.num_timesteps

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    def denoise_step(self, x_t, t, condition, prev_pred=None):
        if "z_mod" not in condition:
            raise ValueError("Squidiff requires a control-derived semantic condition z_mod")
        result = self.diffusion.ddim_sample(
            self.model,
            x_t,
            t,
            clip_denoised=False,
            model_kwargs={"z_mod": condition["z_mod"]},
            eta=self.eta,
        )
        return {"x_prev": result["sample"], "x0_pred": result["pred_xstart"]}
