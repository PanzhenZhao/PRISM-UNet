import math
import warnings
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from scan_strategy import CrossScan_1, CrossScan_2
from scan_strategy import _build_scan3_perm, _build_scan4_perm

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


# =========================================================
# 0) Norm Helper
# =========================================================
def make_norm(c: int, norm: str = "gn"):
    norm = (norm or "gn").lower()
    if norm == "bn":
        return nn.BatchNorm2d(c)
    if norm in ("ln", "ln2d"):
        return nn.GroupNorm(1, c)
    for g in [32, 16, 8, 4, 2, 1]:
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


# =========================================================
# 1) DCT Basis Helpers
# =========================================================
def build_dct_filters(
    dct_h: int,
    dct_w: int,
    u_list: List[int],
    v_list: List[int],
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """(K,H,W) DCT basis filters."""
    K = len(u_list)
    u = torch.tensor(u_list, device=device, dtype=dtype).view(K, 1, 1)
    v = torch.tensor(v_list, device=device, dtype=dtype).view(K, 1, 1)
    x_grid = torch.arange(dct_h, device=device, dtype=dtype).view(1, dct_h, 1)
    y_grid = torch.arange(dct_w, device=device, dtype=dtype).view(1, 1, dct_w)

    def alpha(p, N):
        a = torch.full_like(p, math.sqrt(2.0 / N))
        a[p == 0] = math.sqrt(1.0 / N)
        return a

    au = alpha(u, dct_h)
    av = alpha(v, dct_w)
    cosx = torch.cos(math.pi * u * (x_grid + 0.5) / dct_h)
    cosy = torch.cos(math.pi * v * (y_grid + 0.5) / dct_w)
    return au * av * cosx * cosy


def _map_pairs_7x7_to_s(pairs_7: List[Tuple[int, int]], s: int) -> Tuple[List[int], List[int]]:
    uu, vv = [], []
    for u7, v7 in pairs_7:
        u = int(round(u7 * (s - 1) / 6.0))
        v = int(round(v7 * (s - 1) / 6.0))
        uu.append(u)
        vv.append(v)
    return uu, vv


def _sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


# =========================================================
# 2) Omni-Directional Dynamic MoE (ODD-MoE)
# =========================================================
class ODDMoE(nn.Module):
    """
    Omni-Directional Dynamic MoE (ODD-MoE)

    Changes:
    - safer reshape instead of view
    - floor-based dynamic EMA prior
    - batch-level minimum expert coverage during training
    """

    def __init__(
        self,
        C: int,
        group_count: int = 8,
        dct_hw: int = 16,
        d_state: int = 16,
        expand: int = 2,
        route_temp_init: float = 1.0,
        band_temp_init: float = 1.0,
        spiral_beta: float = 0.5,
        router_hidden: int = 16,
        use_gumbel: bool = True,
        gumbel_tau: float = 1.0,
        use_film: bool = True,
        shuffle_groups: int = 4,
        norm: str = "gn",
        prior_momentum: float = 0.9,
        aux_alpha: float = 1.0,
        min_expert_floor: float = 0.05,
        enforce_min_per_batch: bool = True,
    ):
        super().__init__()
        self._mamba_available = Mamba is not None

        self.C = int(C)
        assert self.C > 0
        self.G = max(1, math.gcd(self.C, max(1, int(group_count))))
        self.Cg = self.C // self.G

        self.dct_hw = int(dct_hw)
        self.spiral_beta = float(spiral_beta)

        self.use_gumbel = bool(use_gumbel)
        self.gumbel_tau = float(gumbel_tau)

        self.prior_momentum = float(prior_momentum)
        self.aux_alpha = float(aux_alpha)
        self.min_expert_floor = float(min_expert_floor)
        self.enforce_min_per_batch = bool(enforce_min_per_batch)

        assert 0.0 <= self.min_expert_floor < 0.2, "min_expert_floor should be in [0, 0.2) for 5 experts."

        # ---------- fixed-K directional prototypes on 7x7 grid ----------
        self._K = 6
        self._pairs_7 = {
            "h": [(6, 0), (5, 0), (4, 0), (6, 1), (5, 1), (4, 1)],
            "v": [(0, 6), (0, 5), (0, 4), (1, 6), (1, 5), (1, 4)],
            "d": [(6, 6), (5, 5), (4, 4), (6, 5), (5, 6), (6, 4)],
        }

        self.band_w_h = nn.Parameter(torch.zeros(self._K))
        self.band_w_v = nn.Parameter(torch.zeros(self._K))
        self.band_w_d = nn.Parameter(torch.zeros(self._K))
        self.band_temp = nn.Parameter(torch.tensor(float(band_temp_init)))
        self.route_temp = nn.Parameter(torch.tensor(float(route_temp_init)))

        # EMA prior over all 5 experts
        self.register_buffer(
            "prior_ema",
            torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float32),
            persistent=False,
        )

        # ---------- shared Mamba ----------
        self.norm_seq = nn.LayerNorm(self.Cg)
        if self._mamba_available:
            self.mamba = Mamba(d_model=self.Cg, d_state=d_state, expand=expand)
        else:
            self.mamba = None

        self.seq_fallback = nn.Sequential(
            nn.Linear(self.Cg, self.Cg, bias=False),
            nn.GELU(),
            nn.Linear(self.Cg, self.Cg, bias=False),
        )
        self._disable_mamba_runtime = False
        self._mamba_warned = False

        self.use_film = bool(use_film)
        if self.use_film:
            self.film_gamma = nn.Parameter(torch.ones(4, self.Cg))
            self.film_beta = nn.Parameter(torch.zeros(4, self.Cg))

        self.conv_expert = nn.Sequential(
            nn.Conv2d(self.Cg, self.Cg, 3, padding=1, groups=self.Cg, bias=False),
            make_norm(self.Cg, norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.Cg, self.Cg, 1, bias=False),
        )

        # 8-dim router
        in_dim = 8
        hid = max(8, int(router_hidden))
        self.router = nn.Sequential(
            nn.Linear(in_dim, hid, bias=False),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Linear(hid, 5, bias=True),
        )

        self.shuffle_groups = int(shuffle_groups) if (self.C % int(shuffle_groups) == 0) else 1
        self.group_pw = nn.Conv2d(self.C, self.C, 1, groups=self.shuffle_groups, bias=False)
        self.fuse = nn.Sequential(
            make_norm(self.C, norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.C, self.C, 1, bias=False),
        )

        self._perm_cache = {}
        self._dct_cache = {}
        self.aux_loss = None

    # ------------------------- utils -------------------------
    @staticmethod
    def _channel_shuffle(x: torch.Tensor, groups: int):
        if groups <= 1:
            return x
        b, c, h, w = x.shape
        x = x.reshape(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
        return x.reshape(b, c, h, w)

    def _get_perm(self, mode: int, H: int, W: int, device: torch.device):
        key = (mode, H, W, device.type, -1 if device.index is None else device.index)
        if key in self._perm_cache:
            return self._perm_cache[key]
        if mode == 3:
            perm, inv = _build_scan3_perm(H, W, device)
        elif mode == 4:
            perm, inv = _build_scan4_perm(H, W, device)
        else:
            raise ValueError("perm only needed for mode 3/4")
        self._perm_cache[key] = (perm, inv)
        return perm, inv

    def _get_dct_filters_cached(self, s_eff: int, device: torch.device) -> Dict[str, torch.Tensor]:
        key = (s_eff, device.type, -1 if device.index is None else device.index)
        if key in self._dct_cache:
            return self._dct_cache[key]
        out = {}
        for name in ["h", "v", "d"]:
            uu, vv = _map_pairs_7x7_to_s(self._pairs_7[name], s_eff)
            out[name] = build_dct_filters(s_eff, s_eff, uu, vv, device=device, dtype=torch.float32)
        self._dct_cache[key] = out
        return out

    def _band_energy_group(self, x_pool_g: torch.Tensor, filt: torch.Tensor, w_param: torch.Tensor) -> torch.Tensor:
        resp = torch.einsum("bgxy,kxy->bgk", x_pool_g.float(), filt).pow(2)
        temp = torch.clamp(self.band_temp, 0.2, 5.0)
        w = torch.softmax(w_param / temp, dim=0).view(1, 1, -1)
        return (resp * w).sum(dim=-1)

    def _structure_coherence_group(self, x_pool_g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        gx = x_pool_g[..., :, 1:] - x_pool_g[..., :, :-1]
        gy = x_pool_g[..., 1:, :] - x_pool_g[..., :-1, :]
        gx = gx[..., :-1, :]
        gy = gy[..., :, :-1]
        J11 = (gx * gx).mean(dim=(-1, -2))
        J22 = (gy * gy).mean(dim=(-1, -2))
        J12 = (gx * gy).mean(dim=(-1, -2))
        disc = (J11 - J22).pow(2) + 4.0 * (J12 * J12)
        return torch.sqrt(disc + eps) / (J11 + J22 + eps)

    # ------------------------- scan wrappers -------------------------
    def _scan_to_seq(self, xg: torch.Tensor, mode: int) -> torch.Tensor:
        B, Cg, H, W = xg.shape
        if mode == 1:
            xs = CrossScan_1.apply(xg)
            return xs[:, 0].transpose(1, 2).contiguous()
        if mode == 2:
            xs = CrossScan_2.apply(xg)
            return xs[:, 0].transpose(1, 2).contiguous()
        if mode == 3:
            perm, _ = self._get_perm(3, H, W, xg.device)
            return xg.flatten(2).index_select(-1, perm).transpose(1, 2).contiguous()
        if mode == 4:
            perm, _ = self._get_perm(4, H, W, xg.device)
            return xg.flatten(2).index_select(-1, perm).transpose(1, 2).contiguous()
        raise ValueError("mode must be 1..4")

    def _seq_to_map(self, seq: torch.Tensor, mode: int, H: int, W: int) -> torch.Tensor:
        B, L, Cg = seq.shape
        y = seq.transpose(1, 2).contiguous()
        if mode == 1:
            return y.reshape(B, Cg, H, W)
        if mode == 2:
            return y.flip(dims=[-1]).reshape(B, Cg, W, H).transpose(2, 3).contiguous()
        if mode == 3:
            _, inv = self._get_perm(3, H, W, seq.device)
            return y.index_select(-1, inv).reshape(B, Cg, H, W)
        if mode == 4:
            _, inv = self._get_perm(4, H, W, seq.device)
            return y.index_select(-1, inv).reshape(B, Cg, H, W)
        raise ValueError("mode must be 1..4")

    def _process_scan_expert(self, xg: torch.Tensor, mode: int, film_id: int) -> torch.Tensor:
        B, Cg, H, W = xg.shape
        seq = self._scan_to_seq(xg, mode=mode)
        seq_in = self.norm_seq(seq)

        if self.mamba is not None and (not self._disable_mamba_runtime):
            try:
                seq_delta = self.mamba(seq_in)
            except Exception as e:
                self._disable_mamba_runtime = True
                if not self._mamba_warned:
                    warnings.warn(f"Mamba runtime failed, fallback used: {repr(e)}", RuntimeWarning)
                    self._mamba_warned = True
                seq_delta = self.seq_fallback(seq_in)
        else:
            seq_delta = self.seq_fallback(seq_in)

        seq = seq + seq_delta
        if self.use_film:
            gamma = self.film_gamma[film_id].view(1, 1, Cg)
            beta = self.film_beta[film_id].view(1, 1, Cg)
            seq = seq * gamma + beta

        return self._seq_to_map(seq, mode=mode, H=H, W=W)

    def _enforce_batch_expert_floor(self, top1: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """
        Ensure each expert gets at least one token in the current batch, when possible.
        top1: (B,G)
        gates: (B,G,5)
        """
        if (not self.training) or (not self.enforce_min_per_batch):
            return top1

        B, G = top1.shape
        BG = B * G
        num_experts = gates.shape[-1]
        if BG < num_experts:
            return top1

        top1_flat = top1.reshape(BG)
        gates_flat = gates.reshape(BG, num_experts)

        used = torch.bincount(top1_flat, minlength=num_experts)
        missing = (used == 0).nonzero(as_tuple=False).squeeze(1)
        if missing.numel() == 0:
            return top1

        confidence = gates_flat.max(dim=-1).values
        candidate_idx = torch.argsort(confidence)  # low-confidence tokens first
        taken = set()

        for expert_id in missing.tolist():
            for idx in candidate_idx.tolist():
                if idx not in taken:
                    top1_flat[idx] = expert_id
                    taken.add(idx)
                    break

        return top1_flat.reshape(B, G)

    # =========================================================
    # Forward
    # =========================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x_groups = x.reshape(B, self.G, self.Cg, H, W)
        xg = x_groups.mean(dim=2)

        s_eff = max(2, min(self.dct_hw, H, W))
        x_pool_g = F.adaptive_avg_pool2d(xg, (s_eff, s_eff))

        filt = self._get_dct_filters_cached(s_eff, x.device)
        Eh = self._band_energy_group(x_pool_g, filt["h"], self.band_w_h)
        Ev = self._band_energy_group(x_pool_g, filt["v"], self.band_w_v)
        Ed = self._band_energy_group(x_pool_g, filt["d"], self.band_w_d)

        t = torch.clamp(self.route_temp, 0.2, 5.0)
        logits3 = torch.stack([Eh, Ev, Ed], dim=-1) / t
        p = torch.softmax(logits3, dim=-1)
        entropy = -(p * (p + 1e-6).log()).sum(dim=-1)
        coh = self._structure_coherence_group(x_pool_g)
        Es = entropy + self.spiral_beta * (1.0 - coh)

        E_dc = x_pool_g.mean(dim=(-1, -2)).abs()

        total_e = Eh + Ev + Ed + 1e-6
        Rh, Rv, Rd = Eh / total_e, Ev / total_e, Ed / total_e

        raw_feat = torch.stack([Eh, Ev, Ed, Es, E_dc], dim=-1)
        feat_norm = (raw_feat - raw_feat.mean(dim=-1, keepdim=True)) / (raw_feat.std(dim=-1, keepdim=True) + 1e-5)

        feat_in = torch.cat(
            [
                torch.sigmoid(feat_norm * 2.0),
                Rh.unsqueeze(-1), Rv.unsqueeze(-1), Rd.unsqueeze(-1),
            ],
            dim=-1,
        )

        logits5 = self.router(feat_in)
        gates = torch.softmax(logits5, dim=-1)

        if self.training and self.use_gumbel:
            tau = max(0.05, float(self.gumbel_tau))
            noisy = (logits5 + _sample_gumbel(logits5.shape, device=logits5.device)) / tau
            top1 = noisy.argmax(dim=-1)
        else:
            top1 = gates.argmax(dim=-1)

        # batch-level minimum coverage
        top1 = self._enforce_batch_expert_floor(top1, gates)

        top1_gate = gates.gather(-1, top1.unsqueeze(-1)).squeeze(-1)

        # -------------------------
        # Dynamic EMA Prior with explicit floor
        # -------------------------
        importance = gates.mean(dim=(0, 1))  # (5,)

        with torch.no_grad():
            e_scan = torch.stack([Eh.mean(), Ev.mean(), Ed.mean(), Es.mean()])
            dyn_scan_target = e_scan / (e_scan.sum() + 1e-8)

            floor = self.min_expert_floor
            target_dist = torch.full((5,), floor, device=gates.device, dtype=gates.dtype)

            remain = 1.0 - 5.0 * floor
            remain = max(remain, 1e-6)

            # 80% of remaining mass to scan experts according to dynamic energy prior
            # 20% of remaining mass to conv expert
            target_dist[:4] += 0.8 * remain * dyn_scan_target
            target_dist[4] += 0.2 * remain

            target_dist = target_dist / target_dist.sum()

            m = float(self.prior_momentum)
            self.prior_ema.mul_(m).add_((1.0 - m) * target_dist)

        prior = self.prior_ema.to(dtype=gates.dtype, device=gates.device)
        imp = importance + 1e-8
        self.aux_loss = self.aux_alpha * (prior * (prior.log() - imp.log())).sum()

        # -------------------------
        # Dispatch
        # -------------------------
        BG = B * self.G
        x_tokens = x_groups.reshape(BG, self.Cg, H, W)
        top1_flat = top1.reshape(BG)
        gate_flat = top1_gate.reshape(BG, 1, 1, 1).to(dtype=x.dtype)

        y_tokens = torch.zeros_like(x_tokens)

        scan_modes = [1, 2, 3, 4]
        for e in range(4):
            idx = (top1_flat == e).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            xe = x_tokens.index_select(0, idx)
            ye = self._process_scan_expert(xe, mode=scan_modes[e], film_id=e)
            ye = ye * gate_flat.index_select(0, idx)
            y_tokens.index_copy_(0, idx, ye)

        idx = (top1_flat == 4).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            xe = x_tokens.index_select(0, idx)
            ye = self.conv_expert(xe)
            ye = ye * gate_flat.index_select(0, idx)
            y_tokens.index_copy_(0, idx, ye)

        merged = y_tokens.reshape(B, C, H, W)
        merged = self._channel_shuffle(merged, groups=self.shuffle_groups)
        merged = self.group_pw(merged)
        delta = self.fuse(merged)

        return x + delta