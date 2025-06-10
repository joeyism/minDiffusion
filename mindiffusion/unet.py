"""
Simple Unet Structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )

        self.is_res = is_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch size, in_channels, 32, 32]

        x = self.main(x)  # [batch size, out_channels, 32, 32]
        if self.is_res:
            x = x + self.conv(x)
            return (
                x / 1.414
            )  # 1/sqrt(2) https://github.com/cloneofsimo/minDiffusion/issues/6#issuecomment-1666951128
        else:
            return self.conv(x)


class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        # if out channels = in channels * 2, there's a maxpool2d(2)
        # so it makes up for it
        super(UnetDown, self).__init__()
        layers = [
            Conv3(in_channels, out_channels),  # [in, out, n, n]
            nn.MaxPool2d(2),  # [in, out, n/2, n/2]
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_size: int = 2) -> None:
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels, out_channels, conv_size, conv_size
            ),  # reverse of pool2d(2)
            Conv3(out_channels, out_channels),
            Conv3(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        if skip is not None:
            x = torch.cat((x, skip), 1)
        x = self.model(x)

        return x


class TimeSiren(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super(TimeSiren, self).__init__()

        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class NaiveUnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_feat: int = 256) -> None:
        super(NaiveUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feat = n_feat

        self.init_conv = Conv3(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())

        self.timeembed = TimeSiren(2 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),  # 4x instead of 2
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Conv2d(2 * n_feat, self.out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [128, 3, 32, 32]
        # t: [128]

        x = self.init_conv(x)  # [128, 128, 32, 32]

        down1 = self.down1(x)  # [128, 128, 16, 16]
        down2 = self.down2(down1)  # [128, 256, 8, 8]
        down3 = self.down3(down2)  # [128, 256, 4, 4]

        thro = self.to_vec(down3)  # [128, 256, 1, 1] set to 1D by the avgpool2d(4)
        # has nonlinearity relu after pooling?

        temb = self.timeembed(t).view(-1, self.n_feat * 2, 1, 1)
        # [128, 256].[128, 256, 1, 1]

        # starting to go back up
        thro = self.up0(thro + temb)  # [128, 256, 4, 4]

        # up part of unet has 2, because it has the skip connection concat 
        up1 = (
            self.up1(thro, down3) + temb
        )  # up1([128, 256, 4, 4], [128, 256, 4, 4]) + [128, 256, 1, 1] -> [128, 256, 8, 8] + [128, 256, 1, 1] -> [128, 256, 8, 8]

        up2 = self.up2(up1, down2)  # [128, 128, 16, 16]
        up3 = self.up3(up2, down1)  # [128, 128, 32, 32]

        _out = torch.cat((up3, x), 1)  # [128, 256, 32, 32]
        out = self.out(_out)  # [128, 3, 32, 32]

        return out


class VAEUnetCifar(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_feat: int = 256) -> None:
        super(VAEUnetCifar, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feat = n_feat

        self.init_conv = Conv3(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.bottleneck = Conv3(2 * n_feat, 2 * n_feat, is_res=True)

        self.timeembed = TimeSiren(2 * n_feat)

        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.up1 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Conv2d(2 * n_feat, self.out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [batch, 8, 4, 4]
        # t: [batch]
        sample = self.init_conv(x)  # [batch, 128, 4, 4]

        down1 = self.down1(sample)  # [batch, 128, 2, 2]
        down2 = self.down2(down1)  # [batch, 256, 1, 1]

        bottleneck = self.bottleneck(down2) # [batch, 256, 1, 1]

        temb = self.timeembed(t).view(-1, self.n_feat * 2, 1, 1)
        # [batch, 256].[batch, 256, 1, 1]

        # starting to go back up
        # Modulation (+) adjusts how existing features should be processed without introducing new semantic dimensions
        # Augmentation (torch.cat) provides new information that expands the semantic space or adds complementary features.
        up2 = self.up2(bottleneck + temb)  # [batch, 128, 2, 2]

        up1 = self.up1(up2, down1)         # [batch, 128, 4, 4]

        out = self.out(torch.cat([up1, sample], dim=1))  # [batch, 4, 4, 4]

        return out

class VAEUnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_feat: int = 256) -> None:
        super(VAEUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feat = n_feat

        self.init_conv = Conv3(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)

        self.bottleneck = Conv3(4 * n_feat, 4 * n_feat, is_res=True)

        self.timeembed = TimeSiren(4 * n_feat)

        self.up3 = UnetUp(4 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up1 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Conv2d(2 * n_feat, self.out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [batch, 4, 32, 32]
        # t: [batch]
        sample = self.init_conv(x)  # [batch, n_feat=512, 32, 32]

        down1 = self.down1(sample)  # [batch, n_feat=512, 16, 16] (same n_feat, lower size)
        down2 = self.down2(down1)  # [batch, n_feat*2=1024, 8, 8]
        down3 = self.down3(down2)  # [batch, n_feat*4=2048, 4, 4]

        bottleneck = self.bottleneck(down3) # [batch, n_feat*4=2048, 4, 4]

        temb = self.timeembed(t).view(-1, self.n_feat * 4, 1, 1)
        # [batch, n_feat*4=2048].[batch, n_feat*4=2048, 1, 1]

        # starting to go back up
        # Modulation (+) adjusts how existing features should be processed without introducing new semantic dimensions
        # Augmentation (torch.cat) provides new information that expands the semantic space or adds complementary features.
        up3 = self.up3(bottleneck + temb)  # [batch, n_feat * 2 = 1024, 8, 8]
        up2 = self.up2(up3, down2)         # [batch, n_feat = 1024, 16, 16]

        up1 = self.up1(up2, down1)         # [batch, n_feat = 512, 32, 32]

        out = self.out(torch.cat([up1, sample], dim=1))  # [batch, 4, 32, 32]

        return out


class CFGVAEUnet(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n_feat: int = 256, text_dim: int=512, num_text_categories: int=10) -> None:
        super(CFGVAEUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feat = n_feat

        self.init_conv = Conv3(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)

        self.bottleneck = Conv3(4 * n_feat, 4 * n_feat, is_res=True)

        self.timeembed = TimeSiren(4 * n_feat)
        
        self.textembed = nn.Sequential(
            nn.Embedding(num_text_categories, text_dim),
            nn.Linear(text_dim, text_dim),
            nn.SiLU(),
            nn.Linear(text_dim, 4*n_feat),
        )

        self.up3 = UnetUp(4 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up1 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Conv2d(2 * n_feat, self.out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, text: torch.Tensor|None=None) -> torch.Tensor:
        # x: [batch, 4, 32, 32]
        # t: [batch]
        sample = self.init_conv(x)  # [batch, n_feat=512, 32, 32]

        down1 = self.down1(sample)  # [batch, n_feat=512, 16, 16] (same n_feat, lower size)
        down2 = self.down2(down1)  # [batch, n_feat*2=1024, 8, 8]
        down3 = self.down3(down2)  # [batch, n_feat*4=2048, 4, 4]

        bottleneck = self.bottleneck(down3) # [batch, n_feat*4=2048, 4, 4]


        temb = self.timeembed(t).view(-1, self.n_feat * 4, 1, 1)
        # [batch, n_feat*4=2048].[batch, n_feat*4=2048, 1, 1]

        if text is not None:
            textemb = self.textembed(text.long()) # [batch, 4*n_feat=2048]
            textemb = textemb.view(-1, self.n_feat*4, 1, 1) # [batch, n_feat*4=2048, 1, 1]
            combined_emb = temb + textemb
        else:
            combined_emb = temb

        # starting to go back up
        # Modulation (+) adjusts how existing features should be processed without introducing new semantic dimensions
        # Augmentation (torch.cat) provides new information that expands the semantic space or adds complementary features.
        up3 = self.up3(bottleneck + combined_emb)  # [batch, n_feat * 2 = 1024, 8, 8]
        up2 = self.up2(up3, down2)         # [batch, n_feat = 1024, 16, 16]

        up1 = self.up1(up2, down1)         # [batch, n_feat = 512, 32, 32]

        out = self.out(torch.cat([up1, sample], dim=1))  # [batch, 4, 32, 32]

        return out
