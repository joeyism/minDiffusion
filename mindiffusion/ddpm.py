from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from diffusers.models import AutoencoderKL


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor: # [512, 3, 32, 32]
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(x.device) # [512]

        # 32x32 -> 4x4 vae
        with torch.no_grad():
            posterior = self.vae.encode(x)
            sample = posterior.latent_dist.sample() * 0.18215
        eps = torch.randn_like(sample)  # eps ~ N(0, 1) but for all of sample, so [512, 8, 4, 4]
        sample_t = (
            self.sqrtab[_ts, None, None, None] * sample
            + self.sqrtmab[_ts, None, None, None] * eps
        )
        pred = self.eps_model(sample_t, _ts / self.n_T, labels) if labels is not None else self.eps_model(sample_t, _ts / self.n_T)

        # 32x32 naive unet
        #eps = torch.randn_like(x)  # eps ~ N(0, 1) [batch, 3, 32, 32]
        #x_t = (
        #    self.sqrtab[_ts, None, None, None] * x
        #    + self.sqrtmab[_ts, None, None, None] * eps
        #) # [batch, 3, 32, 32]
        #pred = self.eps_model(x_t, _ts / self.n_T)

        return self.criterion(eps, pred) #MSELoss

    def sample_vae(self, n_sample: int, size, labels, device) -> torch.Tensor:

        latent_h, latent_w = size[1] // 8, size[2] // 8
        z_i = torch.randn((n_sample, 4, latent_h, latent_w)).to(device)  # [batch, 8, 4, 4] of noise

        # This samples accordingly to Algorithm 2 on page 4. It is exactly the same logic.
        for i in tqdm(range(self.n_T, 0, -1), desc="Sampling"):
            z = torch.randn_like(z_i) if i > 1 else 0

            pred = self.eps_model(
                x=z_i,
                t=torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1),
                text=labels
            )
            z_i = (
                self.oneover_sqrta[i] * (z_i - pred * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        with torch.no_grad():
            return self.vae.decoder(z_i / 0.18215) 

    def sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1) [n_sample, *size]

        # This samples accordingly to Algorithm 2 on page 4. It is exactly the same logic.
        for i in range(self.n_T, 0, -1): # 1000, 999, 998, ..., 3, 2, 1
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0 # [n_sample, *size]
            eps = self.eps_model(
                x_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1)
            ) # [n_sample, *size]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            ) # [n_sample, *size]

        return x_i


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1 # when t = 0, we're at beta1. when t = T, we're at beta 2. linearly
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }
