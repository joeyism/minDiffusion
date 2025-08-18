from torch import nn
import torch

class PatchEmbed(nn.Module):
    """
    Flattens image to patches, and embeds each patch
    """

    def __init__(self, img_size: int, patch_size: int, in_channels: int=3, embed_dim: int=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size)**2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [batch, channels, height, width]
        x = self.proj(x) # [batch, embed_dim=1024, height//patch_size, width//patch_size]
        x = x.flatten(2) # [batch, embed_dim=1024, n_patches]

        x = x.transpose(1, 2) # [batch, n_patches, embed_dim=1024]
        return x


class TimestepEmbedding(nn.Module):

    def __init__(self, dim: int, max_period: int=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        # t: [batch]

        half_dim = self.dim // 2
        # https://claude.ai/public/artifacts/2128ff26-93f0-4ac8-988d-6086e232cb55
        max_period_tensor = torch.tensor(self.max_period, dtype=torch.float32, device=t.device)
        freqs = torch.exp(-torch.log(max_period_tensor) * torch.arange(half_dim, device=t.device) / half_dim)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float=4.0):
        super().__init__()

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6*hidden_size, bias=True)
        )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, bias=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, self.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mlp_hidden_dim, hidden_size)
        )

    def forward(self, x, c):
        # x: [batch, channels, hidden_size]
        # c: [batch, hidden_size]
        modulation_params = self.adaLN_modulation(c) #[batch_size, 6*hidden_size]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation_params.chunk(6, dim=1) # [batch_size, hidden_size] each
        
        norm1 = self.norm1(x)

        norm1_scaled = norm1*(1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(norm1_scaled, norm1_scaled, norm1_scaled)

        x_after_attn = x + gate_msa.unsqueeze(1)*attn_out

        norm2 = self.norm2(x_after_attn)
        norm2_scaled = norm2*(1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(norm2_scaled)
        x_out = x_after_attn + gate_mlp.unsqueeze(1) * mlp_out
        return x_out


class DiT(nn.Module):

    def __init__(self, img_size: int=32, patch_size: int=1, in_channels: int=4, out_channels: int=4, hidden_size: int=768, depth: int=12, num_heads: int=12, mlp_ratio: float=4.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_patches = (img_size // patch_size)**2

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        self.time_embed = TimestepEmbedding(hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.transformer_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6),
            nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        )
        self.final_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2*hidden_size, bias=True)
        )

    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed, std=0.02)

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_normal_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.time_mlp[0].weight, std=0.02)
        torch.nn.init.normal_(self.time_mlp[2].weight, std=0.02)

        for block in self.transformer_blocks:
            torch.nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            torch.nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        torch.nn.init.constant_(self.final_adaLN_modulation[-1].weight, 0)
        torch.nn.init.constant_(self.final_adaLN_modulation[-1].bias, 0)

        torch.nn.init.constant_(self.final_layer[-1].weight, 0)
        torch.nn.init.constant_(self.final_layer[-1].bias, 0)

    def unpatchify(self, x):
        """
        patches to image
        """
        # x: [batch, channels, patch_size * patch_size * out_channels]
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1]**0.5)
        assert h*w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc -> nchpwq", x)
        imgs = x.reshape((x.shape[0], c, h*p, w*p))
        return imgs

    def forward(self, x, t, text):
        # x: [batch, channels, height, width]
        # t: [batch]
        # text: [batch, seq_len, hidden_size] or [batch, hidden_size] if pooled

        x_orig = self.patch_embed(x) # [batch, in_channels, hidden_size]
        x = x_orig + self.pos_embed # add positional embedding

        t_emb = self.time_embed(t) # [batch, hidden_size]
        c = self.time_mlp(t_emb)   # [batch, hidden_size]
        
        # Add text conditioning to time embedding
        if text is not None:
            if text.dim() == 3:  # [batch, seq_len, hidden_size]
                text_pooled = text.mean(dim=1)  # Pool sequence dimension
            else:  # [batch, hidden_size] - already pooled
                text_pooled = text
            c = c + text_pooled  # Combine time and text conditioning

        for block in self.transformer_blocks:
            x = block(x, c)

        shift, scale = self.final_adaLN_modulation(c).chunk(2, dim=1)
        x = self.final_layer[0](x) # layer norm
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.final_layer[1](x)

        img = self.unpatchify(x)
        return img
