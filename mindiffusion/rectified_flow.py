from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from diffusers.models import AutoencoderKL


class RectifiedFlow(nn.Module):
    def __init__(
        self,
        velocity_model: nn.Module,
        n_T: int = 1000,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(RectifiedFlow, self).__init__()
        self.velocity_model = velocity_model
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """
        Rectified Flow training objective.
        Instead of predicting noise, we predict the velocity field v_t = x_1 - x_0.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Random timesteps t ~ Uniform[0, 1]
        t = torch.rand(batch_size, device=device)
        
        # Encode to latent space
        with torch.no_grad():
            posterior = self.vae.encode(x)
            x_1 = posterior.latent_dist.sample() * 0.18215  # x_1 is the real data
        
        # Sample noise as x_0 (starting point)
        x_0 = torch.randn_like(x_1)
        
        # Linear interpolation: x_t = (1-t) * x_0 + t * x_1
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        
        # True velocity: v = x_1 - x_0 (from noise to data)
        v_true = x_1 - x_0
        
        # Predict velocity
        v_pred = self.velocity_model(x_t, t, labels) if labels is not None else self.velocity_model(x_t, t)
        
        return self.criterion(v_pred, v_true)

    def sample_euler(self, n_sample: int, size, labels, device, n_steps: int = 50) -> torch.Tensor:
        """
        Euler sampling method for rectified flow.
        """
        latent_h, latent_w = size[1] // 8, size[2] // 8
        x_t = torch.randn((n_sample, 4, latent_h, latent_w)).to(device)
        
        dt = 1.0 / n_steps
        
        for i in tqdm(range(n_steps), desc="Euler Sampling"):
            t = torch.full((n_sample,), i * dt, device=device)
            
            # Predict velocity
            v_pred = self.velocity_model(
                x=x_t,
                t=t,
                text=labels
            )
            
            # Euler step: x_{t+dt} = x_t + dt * v_pred
            x_t = x_t + dt * v_pred
        
        # Decode from latent space
        with torch.no_grad():
            return self.vae.decoder(x_t / 0.18215)

    def sample_midpoint(self, n_sample: int, size, labels, device, n_steps: int = 50) -> torch.Tensor:
        """
        Midpoint (RK2) sampling method for rectified flow.
        More accurate than Euler method.
        """
        latent_h, latent_w = size[1] // 8, size[2] // 8
        x_t = torch.randn((n_sample, 4, latent_h, latent_w)).to(device)
        
        dt = 1.0 / n_steps
        
        for i in tqdm(range(n_steps), desc="Midpoint Sampling"):
            t = torch.full((n_sample,), i * dt, device=device)
            t_mid = torch.full((n_sample,), (i + 0.5) * dt, device=device)
            
            # First velocity prediction at current point
            v1 = self.velocity_model(
                x=x_t,
                t=t,
                text=labels
            )
            
            # Midpoint estimate
            x_mid = x_t + 0.5 * dt * v1
            
            # Velocity at midpoint
            v2 = self.velocity_model(
                x=x_mid,
                t=t_mid,
                text=labels
            )
            
            # Final step using midpoint velocity
            x_t = x_t + dt * v2
        
        # Decode from latent space
        with torch.no_grad():
            return self.vae.decoder(x_t / 0.18215)

    def sample_rk4(self, n_sample: int, size, labels, device, n_steps: int = 50) -> torch.Tensor:
        """
        4th-order Runge-Kutta sampling method for rectified flow.
        Most accurate but computationally expensive.
        """
        latent_h, latent_w = size[1] // 8, size[2] // 8
        x_t = torch.randn((n_sample, 4, latent_h, latent_w)).to(device)
        
        dt = 1.0 / n_steps
        
        for i in tqdm(range(n_steps), desc="RK4 Sampling"):
            t = torch.full((n_sample,), i * dt, device=device)
            
            # k1
            k1 = self.velocity_model(x_t, t, labels)
            
            # k2
            t_half = torch.full((n_sample,), (i + 0.5) * dt, device=device)
            k2 = self.velocity_model(x_t + 0.5 * dt * k1, t_half, labels)
            
            # k3
            k3 = self.velocity_model(x_t + 0.5 * dt * k2, t_half, labels)
            
            # k4
            t_next = torch.full((n_sample,), (i + 1) * dt, device=device)
            k4 = self.velocity_model(x_t + dt * k3, t_next, labels)
            
            # RK4 update
            x_t = x_t + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Decode from latent space
        with torch.no_grad():
            return self.vae.decoder(x_t / 0.18215)

    def sample_ddim_like(self, n_sample: int, size, labels, device, n_steps: int = 50) -> torch.Tensor:
        """
        DDIM-like deterministic sampling for rectified flow.
        Uses the same deterministic trajectory as DDIM but adapted for rectified flow.
        """
        latent_h, latent_w = size[1] // 8, size[2] // 8
        x_t = torch.randn((n_sample, 4, latent_h, latent_w)).to(device)
        
        # Create time schedule (reverse order for consistency with DDIM)
        times = torch.linspace(1.0, 0.0, n_steps + 1)[:-1]  # [1.0, 0.9, ..., 0.1]
        
        for i, t_val in enumerate(tqdm(times, desc="DDIM-like Sampling")):
            t = torch.full((n_sample,), t_val, device=device)
            
            # Predict velocity
            v_pred = self.velocity_model(x_t, t, labels)
            
            # For rectified flow, we can use the predicted velocity directly
            # to estimate x_0 and then step forward
            dt = 1.0 / n_steps
            x_t = x_t - dt * v_pred  # Step backward in time (since we're going from t=1 to t=0)
        
        # Decode from latent space
        with torch.no_grad():
            return self.vae.decoder(x_t / 0.18215)