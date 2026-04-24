import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.layers import trunc_normal_
except ImportError:
    from torch.nn.init import trunc_normal_

from SSRB_Module import SSRB
try:
    from ODDMOE_Module import ODDMOE
except ImportError:
    from ODDMOE_Module import ODDMoE as ODDMOE
from SPMB_Module import SPMB


def make_stage_norm(c: int):
    for g in [32, 16, 8, 4, 2, 1]:
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


class PRISMUNet(nn.Module):
    """
    PRISM-UNet
    - 6 encoder stages
    - 5 decoder stages
    - 5 SPMB bridge modules
    - 1 final prediction head

    Notes:
    1) final_head is not counted as a decoder block
    2) bridge can be turned off safely for ablation
    3) deep supervision can be turned on/off independently
    """

    def __init__(
        self,
        num_classes: int = 1,
        input_channels: int = 3,
        c_list=(8, 16, 24, 32, 48, 64),
        bridge: bool = True,
        gt_ds: bool = True,
        aux_alpha: float = 1.0,
        upsample_mode: str = "bilinear",
        align_corners: bool = False,
    ):
        super().__init__()

        assert len(c_list) == 6, "c_list must contain 6 channel dimensions."
        self.bridge = bridge
        self.gt_ds = gt_ds
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners
        self.aux_loss_total = None

        c1, c2, c3, c4, c5, c6 = c_list

        # ==================== Encoders ====================
        self.encoder1 = nn.Conv2d(input_channels, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder2 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False)

        # Encoder 3/4: SSRB + 1x1 projection
        self.encoder3 = nn.Sequential(
            SSRB(C=c2, groups=8, norm="gn"),
            nn.Conv2d(c2, c3, kernel_size=1, bias=False),
        )
        self.encoder4 = nn.Sequential(
            SSRB(C=c3, groups=8, norm="gn"),
            nn.Conv2d(c3, c4, kernel_size=1, bias=False),
        )

        # Encoder 5/6: ODDMOE + 1x1 projection
        self.encoder5 = nn.Sequential(
            ODDMOE(C=c4, group_count=8, dct_hw=16, norm="gn", use_gumbel=True, aux_alpha=aux_alpha),
            nn.Conv2d(c4, c5, kernel_size=1, bias=False),
        )
        self.encoder6 = nn.Sequential(
            ODDMOE(C=c5, group_count=8, dct_hw=16, norm="gn", use_gumbel=True, aux_alpha=aux_alpha),
            nn.Conv2d(c5, c6, kernel_size=1, bias=False),
        )

        # ==================== Bridges ====================
        # Always register the attributes so forward is safe.
        if self.bridge:
            self.SPMB1 = SPMB(dim_xh=c2, dim_xl=c1, r=8, detach_mask=True, assume_logits=True, norm="ln2d")
            self.SPMB2 = SPMB(dim_xh=c3, dim_xl=c2, r=8, detach_mask=True, assume_logits=True, norm="ln2d")
            self.SPMB3 = SPMB(dim_xh=c4, dim_xl=c3, r=8, detach_mask=True, assume_logits=True, norm="ln2d")
            self.SPMB4 = SPMB(dim_xh=c5, dim_xl=c4, r=8, detach_mask=True, assume_logits=True, norm="ln2d")
            self.SPMB5 = SPMB(dim_xh=c6, dim_xl=c5, r=8, detach_mask=True, assume_logits=True, norm="ln2d")
        else:
            self.SPMB1 = None
            self.SPMB2 = None
            self.SPMB3 = None
            self.SPMB4 = None
            self.SPMB5 = None

        # ==================== Deep Supervision Heads ====================
        if self.gt_ds:
            self.gt_conv1 = nn.Conv2d(c5, 1, kernel_size=1, bias=True)  # deepest decoder stage, H/32
            self.gt_conv2 = nn.Conv2d(c4, 1, kernel_size=1, bias=True)  # H/16
            self.gt_conv3 = nn.Conv2d(c3, 1, kernel_size=1, bias=True)  # H/8
            self.gt_conv4 = nn.Conv2d(c2, 1, kernel_size=1, bias=True)  # H/4
            self.gt_conv5 = nn.Conv2d(c1, 1, kernel_size=1, bias=True)  # H/2

        # ==================== Decoders ====================
        # Decoder 1/2: ODDMOE + 1x1 projection
        self.decoder1 = nn.Sequential(
            ODDMOE(C=c6, group_count=8, dct_hw=16, norm="gn", use_gumbel=True, aux_alpha=aux_alpha),
            nn.Conv2d(c6, c5, kernel_size=1, bias=False),
        )
        self.decoder2 = nn.Sequential(
            ODDMOE(C=c5, group_count=8, dct_hw=16, norm="gn", use_gumbel=True, aux_alpha=aux_alpha),
            nn.Conv2d(c5, c4, kernel_size=1, bias=False),
        )

        # Decoder 3/4: SSRB + 1x1 projection
        self.decoder3 = nn.Sequential(
            SSRB(C=c4, groups=8, norm="gn"),
            nn.Conv2d(c4, c3, kernel_size=1, bias=False),
        )
        self.decoder4 = nn.Sequential(
            SSRB(C=c3, groups=8, norm="gn"),
            nn.Conv2d(c3, c2, kernel_size=1, bias=False),
        )

        # Decoder 5: plain conv
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c2, c1, kernel_size=3, stride=1, padding=1, bias=False),
        )

        # ==================== Normalization ====================
        # encoder norms
        self.En1 = make_stage_norm(c1)
        self.En2 = make_stage_norm(c2)
        self.En3 = make_stage_norm(c3)
        self.En4 = make_stage_norm(c4)
        self.En5 = make_stage_norm(c5)

        # decoder norms
        self.Dn1 = make_stage_norm(c5)
        self.Dn2 = make_stage_norm(c4)
        self.Dn3 = make_stage_norm(c3)
        self.Dn4 = make_stage_norm(c2)
        self.Dn5 = make_stage_norm(c1)

        # final prediction head
        self.final_head = nn.Conv2d(c1, num_classes, kernel_size=1, bias=True)

        # Only initialize layers defined in THIS file.
        # Do not recursively overwrite custom initialization inside SSRB / ODDMOE / SPMB.
        self._init_local_weights()

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------
    def _init_module(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / n))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_local_weights(self):
        modules_to_init = [
            self.encoder1,
            self.encoder2,
            self.decoder5,
            self.final_head,
            self.En1, self.En2, self.En3, self.En4, self.En5,
            self.Dn1, self.Dn2, self.Dn3, self.Dn4, self.Dn5,
        ]
        if self.gt_ds:
            modules_to_init.extend([
                self.gt_conv1, self.gt_conv2, self.gt_conv3, self.gt_conv4, self.gt_conv5
            ])

        for module in modules_to_init:
            module.apply(self._init_module)

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------
    def _upsample(self, x: torch.Tensor, scale_factor=None, size=None) -> torch.Tensor:
        if self.upsample_mode in ("bilinear", "bicubic", "trilinear"):
            return F.interpolate(
                x,
                scale_factor=scale_factor,
                size=size,
                mode=self.upsample_mode,
                align_corners=self.align_corners,
            )
        return F.interpolate(x, scale_factor=scale_factor, size=size, mode=self.upsample_mode)

    def _apply_bridge(self, bridge_module, xh, xl, mask):
        if bridge_module is None:
            return xl
        return bridge_module(xh, xl, mask)

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------
    def forward(self, x: torch.Tensor):
        # -------------------- Encoder --------------------
        # ---------------------- E1 ----------------------
        out = F.gelu(F.max_pool2d(self.En1(self.encoder1(x)), kernel_size=2, stride=2))
        t1 = out  # (B, c1, H/2,  W/2)
        # ---------------------- E2 ----------------------
        out = F.gelu(F.max_pool2d(self.En2(self.encoder2(out)), kernel_size=2, stride=2))
        t2 = out  # (B, c2, H/4,  W/4)
        # ---------------------- E3 ----------------------
        out = F.gelu(F.max_pool2d(self.En3(self.encoder3(out)), kernel_size=2, stride=2))
        t3 = out  # (B, c3, H/8,  W/8)
        # ---------------------- E4 ----------------------
        out = F.gelu(F.max_pool2d(self.En4(self.encoder4(out)), kernel_size=2, stride=2))
        t4 = out  # (B, c4, H/16, W/16)
        # ---------------------- E5 ----------------------
        out = F.gelu(F.max_pool2d(self.En5(self.encoder5(out)), kernel_size=2, stride=2))
        t5 = out  # (B, c5, H/32, W/32)
        # ---------------------- E6 ----------------------
        out = F.gelu(self.encoder6(out))
        t6 = out  # (B, c6, H/32, W/32)

        # -------------------- Decoder --------------------
        # ---------------------- D5 ----------------------
        out5 = F.gelu(self.Dn1(self.decoder1(t6)))   # (B, c5, H/32, W/32)

        if self.gt_ds:
            gt_logits5 = self.gt_conv1(out5)         # (B,1,H/32,W/32)
            t5_bridge = self._apply_bridge(self.SPMB5, t6, t5, gt_logits5)
            gt_logits5_up = self._upsample(gt_logits5, scale_factor=32)
        else:
            proxy_mask5 = t6.mean(dim=1, keepdim=True)
            t5_bridge = self._apply_bridge(self.SPMB5, t6, t5, proxy_mask5)

        out5 = out5 + t5_bridge

        # ---------------------- D4 ----------------------
        out4 = self.decoder2(out5)                   # (B, c4, H/32, W/32)
        out4 = self._upsample(self.Dn2(out4), scale_factor=2)
        out4 = F.gelu(out4)                          # (B, c4, H/16, W/16)

        if self.gt_ds:
            gt_logits4 = self.gt_conv2(out4)         # (B,1,H/16,W/16)
            t4_bridge = self._apply_bridge(self.SPMB4, t5_bridge, t4, gt_logits4)
            gt_logits4_up = self._upsample(gt_logits4, scale_factor=16)
        else:
            proxy_mask4 = t5_bridge.mean(dim=1, keepdim=True)
            t4_bridge = self._apply_bridge(self.SPMB4, t5_bridge, t4, proxy_mask4)

        out4 = out4 + t4_bridge

        # ---------------------- D3 ----------------------
        out3 = self.decoder3(out4)                   # (B, c3, H/16, W/16)
        out3 = self._upsample(self.Dn3(out3), scale_factor=2)
        out3 = F.gelu(out3)                          # (B, c3, H/8, W/8)

        if self.gt_ds:
            gt_logits3 = self.gt_conv3(out3)         # (B,1,H/8,W/8)
            t3_bridge = self._apply_bridge(self.SPMB3, t4_bridge, t3, gt_logits3)
            gt_logits3_up = self._upsample(gt_logits3, scale_factor=8)
        else:
            proxy_mask3 = t4_bridge.mean(dim=1, keepdim=True)
            t3_bridge = self._apply_bridge(self.SPMB3, t4_bridge, t3, proxy_mask3)

        out3 = out3 + t3_bridge

        # ---------------------- D2 ----------------------
        out2 = self.decoder4(out3)                   # (B, c2, H/8, W/8)
        out2 = self._upsample(self.Dn4(out2), scale_factor=2)
        out2 = F.gelu(out2)                          # (B, c2, H/4, W/4)

        if self.gt_ds:
            gt_logits2 = self.gt_conv4(out2)         # (B,1,H/4,W/4)
            t2_bridge = self._apply_bridge(self.SPMB2, t3_bridge, t2, gt_logits2)
            gt_logits2_up = self._upsample(gt_logits2, scale_factor=4)
        else:
            proxy_mask2 = t3_bridge.mean(dim=1, keepdim=True)
            t2_bridge = self._apply_bridge(self.SPMB2, t3_bridge, t2, proxy_mask2)

        out2 = out2 + t2_bridge

        # ---------------------- D1 ----------------------
        out1 = self.decoder5(out2)                   # (B, c1, H/4, W/4)
        out1 = self._upsample(self.Dn5(out1), scale_factor=2)
        out1 = F.gelu(out1)                          # (B, c1, H/2, W/2)

        if self.gt_ds:
            gt_logits1 = self.gt_conv5(out1)         # (B,1,H/2,W/2)
            t1_bridge = self._apply_bridge(self.SPMB1, t2_bridge, t1, gt_logits1)
            gt_logits1_up = self._upsample(gt_logits1, scale_factor=2)
        else:
            proxy_mask1 = t2_bridge.mean(dim=1, keepdim=True)
            t1_bridge = self._apply_bridge(self.SPMB1, t2_bridge, t1, proxy_mask1)

        out1 = out1 + t1_bridge

        # -------------------- Final Prediction Head --------------------
        out0 = self._upsample(self.final_head(out1), scale_factor=2)

        # -------------------- Collect ODDMOE Auxiliary Regularization --------------------
        aux_list = []
        for blk in [self.encoder5[0], self.encoder6[0], self.decoder1[0], self.decoder2[0]]:
            aux = getattr(blk, "aux_loss", None)
            if aux is not None:
                aux_list.append(aux)
        self.aux_loss_total = sum(aux_list) if len(aux_list) > 0 else None

        if self.gt_ds:
            # deep-to-shallow order
            aux_outputs = (gt_logits5_up, gt_logits4_up, gt_logits3_up, gt_logits2_up, gt_logits1_up)
            return aux_outputs, out0

        return out0