"""
Alignment-Aware Fusion (AAF).

Usage:
    aaf = AlignmentAwareFusion(
            in_ch_stage3_rgb=256, in_ch_stage3_x=256,
            in_ch_stage4_rgb=512, in_ch_stage4_x=512,
            stage5_ch=2048, fused_out_ch=256, bottleneck_ratio=4
        )
    fused = aaf(rgb_s3, x_s3, rgb_s4, x_s4, rgb_s5_hat, x_s5_hat)
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Basic building blocks
# -------------------------
class LocalFusion(nn.Module):
    """
    Local spatial fusion block.
    - Input: X (B, Cin, H, W)
    - Computes:
        Y1 = Conv3x3(Cin -> Cmid)
        Y2 = BN, SiLU
        Y3 = Conv3x3(Cmid -> Cout)
        Y4 = BN
        Shortcut S = BN(Conv1x1(Cin -> Cout))
        out = SiLU(Y4 + S)
    - Typically Cout == Cin to preserve channels; Cmid is configurable.
    """

    def __init__(self, in_channels: int, mid_channels: Optional[int] = None, out_channels: Optional[int] = None, use_bn: bool = True):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels if out_channels is not None else in_channels
        self.mid_ch = mid_channels if mid_channels is not None else max(self.in_ch // 2, 64)
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(self.in_ch, self.mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_ch) if use_bn else nn.Identity()
        self.act = nn.SiLU()

        self.conv2 = nn.Conv2d(self.mid_ch, self.out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_ch) if use_bn else nn.Identity()

        self.shortcut = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(self.out_ch) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Cin, H, W)
        out: (B, Cout, H, W)
        """
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y)

        s = self.shortcut(x)
        s = self.shortcut_bn(s)

        out = self.act(y + s)
        return out


class ChannelReconst(nn.Module):
    """
    Channel-wise reconstruction: per-pixel 1x1 MLP bottleneck.
    - Input: (B, C, H, W)
    - Bottleneck ratio r: C' = max(1, floor(C / r))
    - Conv1x1(C -> C') -> BN -> SiLU -> Conv1x1(C' -> C) -> BN
    - Output: (B, C, H, W)
    """

    def __init__(self, channels: int, bottleneck_ratio: int = 4, use_bn: bool = True):
        super().__init__()
        self.C = channels
        self.r = max(1, bottleneck_ratio)
        self.Cb = max(1, self.C // self.r)
        self.use_bn = use_bn

        self.lin1 = nn.Conv2d(self.C, self.Cb, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.Cb) if use_bn else nn.Identity()
        self.act = nn.SiLU()
        self.lin2 = nn.Conv2d(self.Cb, self.C, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.C) if use_bn else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        out: (B, C, H, W)
        """
        y = self.lin1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.lin2(y)
        y = self.bn2(y)
        return y


class AAFBlock(nn.Module):
    """
    Single AAF block combining LocalFusion and ChannelReconst.
    - Input: tensor U (B, Cin, H, W)
    - Output: (B, Cout, H, W), with residual style: out = U + LocalFusion(U) + ChannelReconst(LocalFusion(U))
    """

    def __init__(self, in_channels: int, mid_channels: Optional[int] = None, out_channels: Optional[int] = None, bottleneck_ratio: int = 4):
        super().__init__()
        out_ch = out_channels if out_channels is not None else in_channels
        self.local = LocalFusion(in_channels, mid_channels=mid_channels, out_channels=out_ch)
        self.chan = ChannelReconst(out_ch, bottleneck_ratio=bottleneck_ratio)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B, Cin, H, W)
        returns: (B, Cout, H, W)
        """
        lf = self.local(u)
        cr = self.chan(lf)
        out = u + lf + cr
        return out


# -------------------------
# Top-level Alignment-Aware Fusion
# -------------------------
class AlignmentAwareFusion(nn.Module):
    """
    Top-down Alignment-Aware Fusion (two-stage).

    Args:
        in_ch_rgb_s3, in_ch_x_s3: channels for stage-3 rgb and x
        in_ch_rgb_s4, in_ch_x_s4: channels for stage-4 rgb and x
        stage5_ch: channel dimension of stage-5 (after CMI), used for upsampled guidance
        fused_out_ch: final desired channel dimension for fused output (typically lower e.g., 256)
        bottleneck_ratio: ratio for ChannelReconst
        mid_ch_factor: factor to control mid channels inside LocalFusion (None -> auto)
    """

    def __init__(
        self,
        in_ch_rgb_s3: int,
        in_ch_x_s3: int,
        in_ch_rgb_s4: int,
        in_ch_x_s4: int,
        stage5_ch: int,
        fused_out_ch: int = 256,
        bottleneck_ratio: int = 4,
        mid_ch_factor: Optional[float] = None,
    ):
        super().__init__()

        # compute input dims for stage4 fusion (concat order: rgb_s4, x_s4, up_rgb5, up_x5)
        in_ch_s4_concat = in_ch_rgb_s4 + in_ch_x_s4 + stage5_ch + stage5_ch
        # compute input dims for stage3 fusion (concat: rgb_s3, x_s3, up(fused_s4))
        in_ch_s3_concat = in_ch_rgb_s3 + in_ch_x_s3 + fused_out_ch

        # choose mid channels for LocalFusion blocks
        def mid_ch_from(in_ch):
            if mid_ch_factor is None:
                return max(64, in_ch // 4)
            return max(16, int(in_ch * mid_ch_factor))

        # AAF block at stage4
        self.stage4_block = AAFBlock(
            in_channels=in_ch_s4_concat,
            mid_channels=mid_ch_from(in_ch_s4_concat),
            out_channels=fused_out_ch,
            bottleneck_ratio=bottleneck_ratio,
        )

        # AAF block at stage3
        self.stage3_block = AAFBlock(
            in_channels=in_ch_s3_concat,
            mid_channels=mid_ch_from(in_ch_s3_concat),
            out_channels=fused_out_ch,
            bottleneck_ratio=bottleneck_ratio,
        )

        # small projection to refine final fused output (optional)
        self.post_proj = nn.Sequential(
            nn.Conv2d(fused_out_ch, fused_out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fused_out_ch),
            nn.SiLU(),
        )

    def _upsample_to(self, src: torch.Tensor, target_hw: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
        """
        Upsample src (B, C, Hs, Ws) to (B, C, Ht, Wt) using bilinear interpolation.
        """
        return F.interpolate(src, size=target_hw, mode=mode, align_corners=False)

    def forward(
        self,
        rgb_s3: torch.Tensor,
        x_s3: torch.Tensor,
        rgb_s4: torch.Tensor,
        x_s4: torch.Tensor,
        rgb_s5_hat: torch.Tensor,
        x_s5_hat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inputs:
            rgb_s3: (B, C_rgb_s3, H3, W3)
            x_s3:   (B, C_x_s3, H3, W3)
            rgb_s4: (B, C_rgb_s4, H4, W4)
            x_s4:   (B, C_x_s4, H4, W4)
            rgb_s5_hat: (B, stage5_ch, H5, W5)
            x_s5_hat:   (B, stage5_ch, H5, W5)

        Returns:
            fused: (B, fused_out_ch, H3, W3) - fused feature map at stage3 resolution
        """
        B = rgb_s4.shape[0]
        # spatial dims
        _, _, H4, W4 = rgb_s4.shape
        _, _, H3, W3 = rgb_s3.shape
        _, _, H5, W5 = rgb_s5_hat.shape

        # 1) Upsample stage5 outputs to stage4 spatial resolution
        up_rgb5_to_4 = self._upsample_to(rgb_s5_hat, target_hw=(H4, W4))
        up_x5_to_4 = self._upsample_to(x_s5_hat, target_hw=(H4, W4))

        # 2) Concatenate modality features at stage4
        # concat order: rgb_s4, x_s4, up_rgb5, up_x5
        cat_s4 = torch.cat([rgb_s4, x_s4, up_rgb5_to_4, up_x5_to_4], dim=1)  # (B, Cin_s4_concat, H4, W4)

        # 3) Apply AAF block at stage4 -> produces fused map at stage4 size (fused_out_ch)
        fused_s4 = self.stage4_block(cat_s4)  # (B, fused_out_ch, H4, W4)

        # 4) Upsample fused_s4 to stage3 resolution
        up_fused_s4_to_3 = self._upsample_to(fused_s4, target_hw=(H3, W3))

        # 5) Concatenate at stage3: rgb_s3, x_s3, up_fused_s4_to_3
        cat_s3 = torch.cat([rgb_s3, x_s3, up_fused_s4_to_3], dim=1)  # (B, Cin_s3_concat, H3, W3)

        # 6) Apply AAF block at stage3 to produce final fused map
        fused_s3 = self.stage3_block(cat_s3)  # (B, fused_out_ch, H3, W3)

        # 7) Refinement projection
        fused = self.post_proj(fused_s3)  # (B, fused_out_ch, H3, W3)
        return fused


# -------------------------
# Smoke test
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example dims (match typical ResNet features)
    B = 2
    H3, W3 = 64, 64    # stage3 spatial
    H4, W4 = 32, 32    # stage4 spatial
    H5, W5 = 16, 16    # stage5 spatial

    C_rgb_s3 = 256
    C_x_s3 = 256
    C_rgb_s4 = 512
    C_x_s4 = 512
    stage5_ch = 2048
    fused_out_ch = 256

    aaf = AlignmentAwareFusion(
        in_ch_rgb_s3=C_rgb_s3, in_ch_x_s3=C_x_s3,
        in_ch_rgb_s4=C_rgb_s4, in_ch_x_s4=C_x_s4,
        stage5_ch=stage5_ch, fused_out_ch=fused_out_ch,
        bottleneck_ratio=4
    ).to(device)

    rgb_s3 = torch.randn(B, C_rgb_s3, H3, W3, device=device)
    x_s3 = torch.randn(B, C_x_s3, H3, W3, device=device)
    rgb_s4 = torch.randn(B, C_rgb_s4, H4, W4, device=device)
    x_s4 = torch.randn(B, C_x_s4, H4, W4, device=device)
    rgb_s5_hat = torch.randn(B, stage5_ch, H5, W5, device=device)
    x_s5_hat = torch.randn(B, stage5_ch, H5, W5, device=device)

    with torch.no_grad():
        fused = aaf(rgb_s3, x_s3, rgb_s4, x_s4, rgb_s5_hat, x_s5_hat)

    print("Fused shape:", fused.shape)
    assert fused.shape == (B, fused_out_ch, H3, W3)
    print("AAF smoke test passed.")
