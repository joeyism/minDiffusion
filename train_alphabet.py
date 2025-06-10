from typing import Dict, Optional, Tuple
from sympy import Ci
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from syntht2i import ShapeDataset

from mindiffusion.unet import NaiveUnet, VAEUnet
from mindiffusion.ddpm import DDPM


def train_alphabet(
    n_epoch: int = 300, device: str = "cuda:1", load_pth: Optional[str] = None
) -> None:

    #eps_model = NaiveUnet(3, 3, n_feat=128)
    eps_model = VAEUnet(4, 4, n_feat=512)

    ddpm = DDPM(
        eps_model=eps_model,
        betas=(1e-4, 0.02),
        n_T=1000
    )
    for param in ddpm.vae.parameters():
        param.requires_grad = False
    ddpm.vae.eval()

    if load_pth is not None:
        ddpm.load_state_dict(torch.load("ddpm_cifar.pth"))

    ddpm.to(device)

    dataset = ShapeDataset(
        length=1000,        # Number of images
        image_size=256,     # Image size (square)
        max_shapes=3,       # Maximum shapes per image
        seed=42,            # Random seed for reproducibility
        nocolor=True      # White Background
    )


    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=16)
    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epoch, eta_min=1e-6)

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            B, C, H, W = x.shape
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        if i % 10 == 0:
            ddpm.eval()
            with torch.no_grad():
                xh = ddpm.sample_vae(4, (C, H, W), device)
                xset = torch.cat([xh, x[:4]], dim=0)
                grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
                save_image(grid, f"./contents/ddpm_sample_cifar{i}.png")

            # save model
            #torch.save(ddpm.state_dict(), f"./ddpm_cifar.pth")


if __name__ == "__main__":
    train_alphabet()
