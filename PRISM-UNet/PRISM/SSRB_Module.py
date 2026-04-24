import math
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1) Helpers
# =========================================================
def make_norm(c: int, norm: str = "gn"):
    norm = (norm or "gn").lower()
    if norm == "bn":
        return nn.BatchNorm2d(c)
    if norm in ("ln", "ln2d"):
        return nn.GroupNorm(1, c)  # LN-like for conv feats
    # default GN
    for g in [32, 16, 8, 4, 2, 1]:
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


def build_dct_flat_t(dct_h: int, dct_w: int, uv_list: List[Tuple[int, int]], device=None):
    """
    uv_list: list of (u,v) indices in [0..dct_h-1], [0..dct_w-1]
    return: (HW, K) fp32
    """
    K = len(uv_list)
    u = torch.tensor([u for u, v in uv_list], device=device, dtype=torch.float32).view(K, 1, 1)
    v = torch.tensor([v for u, v in uv_list], device=device, dtype=torch.float32).view(K, 1, 1)
    x = torch.arange(dct_h, device=device, dtype=torch.float32).view(1, dct_h, 1)
    y = torch.arange(dct_w, device=device, dtype=torch.float32).view(1, 1, dct_w)

    def alpha(p, N):
        a = torch.full_like(p, math.sqrt(2.0 / N))
        a[p == 0] = math.sqrt(1.0 / N)
        return a

    cu = alpha(u, dct_h)
    cv = alpha(v, dct_w)
    basis = cu * cv * torch.cos(math.pi * u * (x + 0.5) / dct_h) * torch.cos(math.pi * v * (y + 0.5) / dct_w)  # (K,H,W)
    flat_t = basis.reshape(K, -1).t().contiguous()  # (HW, K)
    return flat_t  # fp32


def radial_uv(dct_h: int, dct_w: int, band: str, k: int, exclude_dc: bool = True):
    """
    Simple radial band selection in frequency plane (u,v).
    radius = u^2 + v^2
    exclude_dc: if True, low-band will NOT include (0,0), avoiding gate dominated by DC/mean.
    """
    uv = [(u, v) for u in range(dct_h) for v in range(dct_w)]
    uv.sort(key=lambda t: (t[0] ** 2 + t[1] ** 2))

    if exclude_dc and band == "low":
        uv = [(u, v) for (u, v) in uv if not (u == 0 and v == 0)]
        if len(uv) == 0:
            uv = [(0, 0)]

    k = max(1, min(int(k), len(uv)))

    if band == "low":
        sel = uv[:k]
    elif band == "high":
        sel = uv[-k:]
    else:  # mid
        mid = len(uv) // 2
        half = k // 2
        start = max(0, min(len(uv) - k, mid - half))
        sel = uv[start:start + k]
    return sel


# =========================================================
# 2) Spectral-Spatial Routing Block (SSRB)
# =========================================================
class SSRB(nn.Module):
    """
    Spectral-Spatial Routing Block

    Features:
    - low-band default excludes DC (0,0) to prevent gating from being dominated by mean/brightness.
    - Energy calculated via squares (coef^2) for robustness.
    - Anti-aliasing maintained: no upsampling prior to DCT.
    - Full-resolution dilated branches.
    """

    def __init__(
            self,
            C: int,
            groups: int = 8,
            freq_sizes: Tuple[int, ...] = (16, 8, 4),
            dilations: Tuple[int, ...] = (1, 2, 3),
            k_per_band: int = 6,
            norm: str = "gn",
            exclude_dc: bool = True,
    ):
        super().__init__()
        assert C > 0
        self.C = C
        self.norm = norm

        G = math.gcd(C, max(1, groups))
        self.G = max(1, G)

        self.freq_sizes = tuple(freq_sizes)
        self.dilations = tuple(dilations)
        self.k_per_band = int(k_per_band)
        self.exclude_dc = bool(exclude_dc)

        S = len(self.freq_sizes)
        assert len(self.dilations) == S, "len(dilations) must match len(freq_sizes)"

        # descriptor dim = (S scales) * (3 bands) * (G groups)
        desc_dim = self.G * S * 3
        hidden = max(16, C // 8)

        self.mlp = nn.Sequential(
            nn.Linear(desc_dim, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.to_g_group = nn.Sequential(nn.Linear(hidden, self.G, bias=True), nn.Sigmoid())
        self.to_a_scale = nn.Linear(hidden, S, bias=True)

        # full-res dilated branches
        self.dw = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(C, C, 3, padding=d, dilation=d, groups=C, bias=False),
                make_norm(C, norm),
                nn.ReLU(inplace=True),
            ) for d in self.dilations
        ])

        self.fuse =nn.Conv2d(C, C, 1, bias=False)


        # Python cache for DCT bases
        self._dct_cache: Dict[Tuple[int, str, str], torch.Tensor] = {}

    def _device_key(self, device: torch.device) -> str:
        if device.type == "cuda":
            return f"cuda:{device.index}"
        return "cpu"

    def _get_dct_flat_all(self, s_eff: int, device: torch.device) -> Tuple[torch.Tensor, int]:
        """
        Return concatenated bases: flat_all = [low|mid|high] => (HW, 3K), fp32 on `device`.
        Also returns K.
        """
        devk = self._device_key(device)
        cache_key = (s_eff, "all", devk)
        if cache_key in self._dct_cache:
            flat_all = self._dct_cache[cache_key]
            K = flat_all.shape[1] // 3
            return flat_all, K

        uv_low = radial_uv(s_eff, s_eff, "low", self.k_per_band, exclude_dc=self.exclude_dc)
        uv_mid = radial_uv(s_eff, s_eff, "mid", self.k_per_band, exclude_dc=False)
        uv_high = radial_uv(s_eff, s_eff, "high", self.k_per_band, exclude_dc=False)

        K_common = max(1, min(len(uv_low), len(uv_mid), len(uv_high)))
        uv_low = uv_low[:K_common]
        uv_mid = uv_mid[:K_common]
        uv_high = uv_high[:K_common]

        flat_low = build_dct_flat_t(s_eff, s_eff, uv_low, device=device)
        flat_mid = build_dct_flat_t(s_eff, s_eff, uv_mid, device=device)
        flat_high = build_dct_flat_t(s_eff, s_eff, uv_high, device=device)

        flat_all = torch.cat([flat_low, flat_mid, flat_high], dim=1).contiguous()  # (HW,3K)
        self._dct_cache[cache_key] = flat_all
        return flat_all, K_common

    def _freq_energy(self, xg: torch.Tensor, target_s: int) -> torch.Tensor:
        """
        xg: (B,G,H,W) -> pool to (s_eff,s_eff) -> matmul with bases -> energy (B,G,3)
        """
        B, G, H, W = xg.shape
        s_eff = min(int(target_s), int(H), int(W))
        s_eff = max(2, s_eff)  # avoid degenerate 1x1 DCT

        xp = F.adaptive_avg_pool2d(xg, (s_eff, s_eff))  # (B,G,s,s)
        xf = xp.reshape(B, G, -1).to(torch.float32)  # (B,G,HW) fp32

        flat_all, K = self._get_dct_flat_all(s_eff, xg.device)
        coef = torch.matmul(xf, flat_all)

        # energy: square
        coef2 = coef * coef
        coef2 = coef2.view(B, G, 3, K)

        e = torch.log1p(coef2).mean(dim=-1)
        return e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == self.C
        G = self.G

        xg = x.view(B, G, C // G, H, W).mean(dim=2)

        desc_parts = []
        for s in self.freq_sizes:
            e = self._freq_energy(xg, s)
            desc_parts.append(e.reshape(B, -1))
        desc = torch.cat(desc_parts, dim=1)

        h = self.mlp(desc)
        g_group = self.to_g_group(h)
        a_scale = F.softmax(self.to_a_scale(h), dim=-1)

        # --- Step 1: Channel Gating  ---
        g_ch = g_group.repeat_interleave(self.C // self.G, dim=1).to(dtype=x.dtype)
        x_gated = x * g_ch.view(B, C, 1, 1)

        # --- Step 2: Spatial Multi-scale Feature Extraction ---
        y = torch.zeros_like(x)
        for i in range(len(self.dilations)):
            yi = self.dw[i](x_gated)  # 注意这里用的是 x_gated
            w = a_scale[:, i].to(dtype=x.dtype).view(B, 1, 1, 1)
            y = y + w * yi

        # --- Step 3: Final Fuse ---
        out = self.fuse(y)
        return x + out



