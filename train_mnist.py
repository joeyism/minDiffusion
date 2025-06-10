import random
import os
from typing import Dict, Optional, Tuple
from sympy import Ci
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from datasets import load_dataset

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from syntht2i import ShapeDataset

from mindiffusion.unet import NaiveUnet, VAEUnet, CFGVAEUnet
from mindiffusion.ddpm import DDPM


class ImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item.get("label", -1)

        if self.transform:
            image = self.transform(image)

        return image, label


def to_rgb(image):
    return image.convert("RGB")


def train_mnist(
    n_epoch: int = 200,
    device: str = "cuda:1",
    load_path: str = "./contents/ddpm_cifar.pth",
) -> None:

    # eps_model = NaiveUnet(3, 3, n_feat=128)
    #eps_model = VAEUnet(4, 4, n_feat=512) 
    eps_model = CFGVAEUnet(4, 4, n_feat=512, text_dim=10)

    ddpm = DDPM(eps_model=eps_model, betas=(1e-4, 0.02), n_T=1000)

    if os.path.exists(load_path):
        print(f"{load_path} found, loading now")
        weights = torch.load(load_path)
        ddpm.load_state_dict(weights)

    for param in ddpm.vae.parameters():
        param.requires_grad = False
    ddpm.vae.eval()

    ddpm.to(device)

    # dataset = load_dataset("imagefolder", data_dir="/root/data/256x256")

    random.seed(42)
    torch.manual_seed(42)
    dataset = load_dataset("joeyism/mnist-256x256", revision="refs/convert/parquet")

    datasets = {
        label: dataset["validation"].filter(lambda x: x["label"] == label)
        for label in set(dataset["validation"]["label"])
    }
    shortest_labels_length = min(len(val) for val in datasets.values())
    shortest_labels_length = (shortest_labels_length // 8 - 1)*8
    subset = [
        Subset(_ds, indices=list(range(shortest_labels_length)))
        for _ds in datasets.values()
    ]
    concat_ds = ConcatDataset(subset)
    train_dataset = ImageDataset(
        concat_ds,
        transform=transforms.Compose(
            [
                transforms.Lambda(to_rgb),
                transforms.ToTensor(),
            ]
        ),
    )

    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=15)
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5, weight_decay=1e-6)
    # scheduler = None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.1, patience=5
    )

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, labels in pbar:
            B, C, H, W = x.shape
            optim.zero_grad()
            x = x.to(device)
            labels = labels.to(device, dtype=x.dtype).reshape(-1, 1) #[batch, 1]
            #loss = ddpm(x) # non cfg
            loss = ddpm(x, labels) # cfg
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=1.0)
            optim.step()

        if i % 10 == 0:
            ddpm.eval()
            with torch.no_grad():
                xh = ddpm.sample_vae(4, (C, H, W), labels=labels[:4], device=device)
                xset = torch.cat([xh, x[:4]], dim=0)
                grid = make_grid(xset, normalize=True, value_range=(0, 1), nrow=4)
                save_image(grid, f"./contents/ddpm_sample_cifar{i}.png")

            # save model
            torch.save(ddpm.state_dict(), load_path)

        if loss_ema and scheduler:
            scheduler.step(
                loss_ema
            )  # adjust learning rate based on the validation loss


if __name__ == "__main__":
    train_mnist()
