import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """
    LayerNorm over channel dimension for 2D feature maps.
    For each spatial location, normalize across channels.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MaskPriorExtractor(nn.Module):
    """
    Extracts [probability, boundary, uncertainty] from mask predictions.
    Input: mask_logits or prob. Default assumes logits for stability.
    Output: (B, 3, H, W)
    """

    def __init__(self, detach_mask=True, assume_logits=True):
        super().__init__()
        self.detach_mask = detach_mask
        self.assume_logits = assume_logits

    def forward(self, mask):
        p = torch.sigmoid(mask) if self.assume_logits else mask
        if self.detach_mask:
            p = p.detach()

        # Uncertainty: u in [0,1], peak at p=0.5
        u = 4.0 * p * (1.0 - p)

        # Boundary: Morphological gradient via max pooling
        max_p = F.max_pool2d(p, kernel_size=3, stride=1, padding=1)
        min_p = -F.max_pool2d(-p, kernel_size=3, stride=1, padding=1)
        b = (max_p - min_p).clamp(0, 1)

        return torch.cat([p, b, u], dim=1)


class SPMB(nn.Module):
    """
    Synergistic Prior Modulation Bridge (SPMB)

    A lightweight, prior-guided UNet bridge featuring:
    (1) Pixel-wise fusion gate (xh vs xl) tempered by uncertainty
    (2) Multi-RF DWConv mixture (d=1/2/3) steered by spatial priors
    (3) Low-rank FiLM modulation derived from a global prior token
    """

    def __init__(
            self,
            dim_xh: int,
            dim_xl: int,
            dim_out: int = None,
            r: int = 8,  # low-rank projection dimension
            detach_mask: bool = True,
            assume_logits: bool = True,
            norm: str = "ln2d",
    ):
        super().__init__()
        self.dim_xl = dim_xl
        self.dim_out = dim_out or dim_xl

        # Align high-res channels to low-res
        self.align_xh = nn.Conv2d(dim_xh, dim_xl, 1, bias=False)

        # Spatial priors extractor [p, b, u]
        self.prior_extractor = MaskPriorExtractor(detach_mask=detach_mask, assume_logits=assume_logits)

        # (A) Pixel-wise fusion gate: s(x,y) in [0,1]
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(8, 1, 1, bias=True),
        )

        # (B) Multi-RF router weights: w(x,y) over 3 branches
        self.rf_router = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(8, 3, 1, bias=True),
        )

        # Depthwise parallel branches for varying receptive fields
        self.dw1 = nn.Conv2d(dim_xl, dim_xl, 3, padding=1, dilation=1, groups=dim_xl, bias=False)
        self.dw2 = nn.Conv2d(dim_xl, dim_xl, 3, padding=2, dilation=2, groups=dim_xl, bias=False)
        self.dw3 = nn.Conv2d(dim_xl, dim_xl, 3, padding=3, dilation=3, groups=dim_xl, bias=False)

        self.norm2d = LayerNorm2d(dim_xl) if norm in ("ln2d", "ln") else nn.GroupNorm(8, dim_xl)

        # (C) Low-rank FiLM from global prior token
        self.mlp = nn.Sequential(
            nn.Linear(3, max(8, r), bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, r), r, bias=True),
        )
        self.Wg = nn.Parameter(torch.zeros(r, dim_xl))
        self.Wb = nn.Parameter(torch.zeros(r, dim_xl))

        self.tail_proj = nn.Conv2d(dim_xl, self.dim_out, 1, bias=False)
        self.skip_proj = nn.Identity() if self.dim_out == dim_xl else nn.Conv2d(dim_xl, self.dim_out, 1, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize fusion gate near identity/zero
        nn.init.zeros_(self.fuse_gate[-1].weight)
        nn.init.zeros_(self.fuse_gate[-1].bias)

        # Bias the RF router to favor the standard 3x3 DWConv initially
        nn.init.zeros_(self.rf_router[-1].weight)
        self.rf_router[-1].bias.data[:] = torch.tensor([1.0, 0.0, 0.0])

        # Initialize LoRA projections to zero (acts as identity mapping initially)
        nn.init.zeros_(self.Wg)
        nn.init.zeros_(self.Wb)

    def forward(self, xh, xl, mask):
        B, C, H, W = xl.shape

        xh = self.align_xh(xh)
        xh = F.interpolate(xh, size=(H, W), mode="bilinear", align_corners=False)

        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)

        priors = self.prior_extractor(mask)  # [p, b, u] -> (B, 3, H, W)
        u = priors[:, 2:3]

        # ----- (A) Pixel-wise Fusion Gate -----
        s = torch.sigmoid(self.fuse_gate(priors))
        # Uncertainty tempering: where u is high, push s towards 0.5 (blend evenly)
        s = (1.0 - u) * s + u * 0.5
        fused = s * xh + (1.0 - s) * xl

        # ----- (B) Multi-RF DWConv Mixture -----
        w = torch.softmax(self.rf_router(priors), dim=1)
        y1 = self.dw1(fused)
        y2 = self.dw2(fused)
        y3 = self.dw3(fused)
        feat = w[:, 0:1] * y1 + w[:, 1:2] * y2 + w[:, 2:3] * y3
        feat = self.norm2d(feat)

        # ----- (C) Low-rank Global FiLM -----
        g = priors.mean(dim=(-1, -2))  # Aggregate priors -> (B, 3)
        a = self.mlp(g)  # Intermediate rep -> (B, r)
        gamma = a @ self.Wg  # Learnable scaling -> (B, C)
        beta = a @ self.Wb  # Learnable shifting -> (B, C)
        feat = feat * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

        out = self.tail_proj(feat)
        return self.skip_proj(xl) + out