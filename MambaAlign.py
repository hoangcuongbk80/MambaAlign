import warnings
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from rgbx_backbone import RGBXBackbone
from pmm_qsvss import PerModalMamba
from cross_mamba_cmi import CrossMambaInteraction
from alignment_aware_fusion import AlignmentAwareFusion

SuperSimpleNet = None
try:
    # common naming possibilities
    from SuperSimpleNet.model import SuperSimpleNet as SuperSimpleNet_class  # repo may use this
    SuperSimpleNet = SuperSimpleNet_class
except Exception:
    try:
        from super_simple_net import SuperSimpleNet as SuperSimpleNet_class
        SuperSimpleNet = SuperSimpleNet_class
    except Exception:
        try:
            from supersimplenet import SuperSimpleNet as SuperSimpleNet_class
            SuperSimpleNet = SuperSimpleNet_class
        except Exception:
            SuperSimpleNet = None


class SimpleDecoder(nn.Module):
    """
    Small convolutional decoder used as fallback if SuperSimpleNet is not available.
    Expects fused features at stage3 spatial resolution (H3, W3).
    Produces segmentation logits at input image resolution (upsampled).
    """

    def __init__(self, in_ch: int, mid_ch: int = 128, n_classes: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.logits = nn.Conv2d(mid_ch, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, out_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        x: (B, in_ch, H3, W3)
        out_hw: optional target spatial size (H_orig, W_orig) to upsample logits
        returns: logits (B, 1, H_out, W_out) where H_out = out_hw[0] or x.shape[-2]
        """
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y)
        logits = self.logits(y)
        if out_hw is not None:
            logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
        return logits


class MambaAlign(nn.Module):
    """
    Main MambaAlign model combining:
      - RGBXBackbone (shared convs + modality BNs)
      - PerModalMamba applied to stage3, stage4, stage5 for both modalities
      - CrossMambaInteraction at stage5
      - AlignmentAwareFusion top-down fusion
      - Decoder (SuperSimpleNet if available; fallback SimpleDecoder)
      - Image-level classifier head

    Forward API:
        logits, score, aux = model(rgb, x)
        logits: (B, 1, H, W) - segmentation logits upsampled to input resolution
        score: (B,) - image-level anomaly score (sigmoid)
        aux: dict of intermediate tensors (optional)
    """

    def __init__(
        self,
        x_in_ch: int = 1,
        pretrained_backbone: bool = True,
        backbone_name: str = "wide_resnet50_2",
        pmm_blocks_per_stage: Tuple[int, int, int] = (4, 4, 4),
        pmm_route_factor: int = 4,
        cmi_D: int = 256,
        cmi_N: int = 128,
        fused_out_ch: int = 256,
        decoder_name: Optional[str] = "super_simple_net",
        use_super_simple_net: bool = True,
        classifier_hidden: int = 256,
    ):
        super().__init__()

        # 1) backbone
        self.backbone = RGBXBackbone(x_in_ch=x_in_ch, pretrained=pretrained_backbone, backbone_name=backbone_name)

        # Stage channels — infer by doing a dummy pass or assume defaults from WideResNet50
        # Default wide_resnet50_2 layers shapes:
        # layer2 -> 512, layer3 -> 1024, layer4 -> 2048
        # But keep these as configurable / deduced later.
        self.stage3_ch = 512
        self.stage4_ch = 1024
        self.stage5_ch = 2048

        # 2) PMM per-modality per-stage: we create separate PerModalMamba instances for each stage and modality
        # Note: PerModalMamba expects `in_ch` channel size
        # We will name them: pmm_rgb_s3, pmm_x_s3, pmm_rgb_s4, ...
        s3_blocks, s4_blocks, s5_blocks = pmm_blocks_per_stage
        self.pmm_rgb_s3 = PerModalMamba(in_ch=self.stage3_ch, num_blocks=s3_blocks, route_factor=pmm_route_factor)
        self.pmm_x_s3 = PerModalMamba(in_ch=self.stage3_ch, num_blocks=s3_blocks, route_factor=pmm_route_factor)

        self.pmm_rgb_s4 = PerModalMamba(in_ch=self.stage4_ch, num_blocks=s4_blocks, route_factor=pmm_route_factor)
        self.pmm_x_s4 = PerModalMamba(in_ch=self.stage4_ch, num_blocks=s4_blocks, route_factor=pmm_route_factor)

        self.pmm_rgb_s5 = PerModalMamba(in_ch=self.stage5_ch, num_blocks=s5_blocks, route_factor=pmm_route_factor)
        self.pmm_x_s5 = PerModalMamba(in_ch=self.stage5_ch, num_blocks=s5_blocks, route_factor=pmm_route_factor)

        # 3) Cross Mamba Interaction at stage5
        self.cmi = CrossMambaInteraction(in_ch=self.stage5_ch, D=cmi_D, N=cmi_N)

        # 4) Alignment Aware Fusion
        self.aaf = AlignmentAwareFusion(
            in_ch_rgb_s3=self.stage3_ch,
            in_ch_x_s3=self.stage3_ch,
            in_ch_rgb_s4=self.stage4_ch,
            in_ch_x_s4=self.stage4_ch,
            stage5_ch=self.stage5_ch,
            fused_out_ch=fused_out_ch,
            bottleneck_ratio=4,
        )

        # 5) Decoder: prefer SuperSimpleNet if available in repository; else fallback
        self.fused_out_ch = fused_out_ch
        if use_super_simple_net and SuperSimpleNet is not None:
            try:
                # Common SuperSimpleNet constructors accept "in_channels" or similar.
                # Try several common constructor signatures to be robust.
                try:
                    self.decoder = SuperSimpleNet(in_channels=self.fused_out_ch, num_classes=1)
                except TypeError:
                    # alternative naming
                    self.decoder = SuperSimpleNet(num_classes=1, in_ch=self.fused_out_ch)
            except Exception as e:
                warnings.warn(f"SuperSimpleNet found but failed to construct: {e}. Falling back to SimpleDecoder.")
                self.decoder = SimpleDecoder(in_ch=self.fused_out_ch, mid_ch=128, n_classes=1)
        else:
            self.decoder = SimpleDecoder(in_ch=self.fused_out_ch, mid_ch=128, n_classes=1)

        # 6) Image-level classifier head (global pooling -> MLP -> sigmoid)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.fused_out_ch, classifier_hidden),
            nn.SiLU(),
            nn.Linear(classifier_hidden, 1),
        )

        # small weight init for classifier
        nn.init.xavier_uniform_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, rgb: torch.Tensor, x: torch.Tensor, return_aux: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass through the entire MambaAlign model.

        Inputs:
            rgb: (B, 3, H, W)
            x:   (B, x_in_ch, H, W)

        Outputs:
            logits: (B, 1, H, W) segmentation logits (not sigmoid-ed)
            score:  (B,) image-level anomaly score (sigmoid activation applied)
            aux:    dict with intermediate tensors (optional)
        """
        B, _, H_in, W_in = rgb.shape

        # 1) backbone
        feats = self.backbone(rgb, x)
        # feats keys: rgb_stage3, rgb_stage4, rgb_stage5, x_stage3, x_stage4, x_stage5
        rgb_s3 = feats["rgb_stage3"]
        rgb_s4 = feats["rgb_stage4"]
        rgb_s5 = feats["rgb_stage5"]
        x_s3 = feats["x_stage3"]
        x_s4 = feats["x_stage4"]
        x_s5 = feats["x_stage5"]

        aux = {"backbone_shapes": {k: v.shape for k, v in feats.items()}}

        # 2) Per-modal PMM on each stage
        rgb_s3_p = self.pmm_rgb_s3(rgb_s3)
        x_s3_p = self.pmm_x_s3(x_s3)

        rgb_s4_p = self.pmm_rgb_s4(rgb_s4)
        x_s4_p = self.pmm_x_s4(x_s4)

        rgb_s5_p = self.pmm_rgb_s5(rgb_s5)
        x_s5_p = self.pmm_x_s5(x_s5)

        aux["pmm_shapes"] = {
            "rgb_s3_p": rgb_s3_p.shape, "x_s3_p": x_s3_p.shape,
            "rgb_s4_p": rgb_s4_p.shape, "x_s4_p": x_s4_p.shape,
            "rgb_s5_p": rgb_s5_p.shape, "x_s5_p": x_s5_p.shape,
        }

        # 3) Cross Mamba Interaction (CMI) at stage5 -> cross-enhanced stage5 features
        rgb_s5_hat, x_s5_hat, cmi_info = self.cmi(rgb_s5_p, x_s5_p)
        aux["cmi_info"] = {k: v.shape if isinstance(v, torch.Tensor) else None for k, v in cmi_info.items()}

        # 4) Alignment-Aware Fusion (top-down)
        fused = self.aaf(rgb_s3_p, x_s3_p, rgb_s4_p, x_s4_p, rgb_s5_hat, x_s5_hat)
        aux["fused_shape"] = fused.shape

        # 5) Decoder -> segmentation logits upsampled to input image resolution
        # decoder forward expects fused features at stage3 resolution; ensure we upsample logits to (H_in, W_in)
        logits = self.decoder(fused, out_hw=(H_in, W_in))  # (B,1,H_in,W_in) or (B, n_classes,...)
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        elif logits.dim() == 4 and logits.shape[1] != 1:
            # If decoder outputs multi-class logits, reduce to single anomaly logit via a linear head
            # But typically decoder outputs 1 channel for anomaly segmentation.
            pass

        # 6) Image-level classifier on fused global pooled feature
        pooled = F.adaptive_avg_pool2d(fused, 1).view(B, -1)  # (B, fused_out_ch)
        score_logit = self.classifier(pooled).view(B)  # (B,)
        score = torch.sigmoid(score_logit)

        if return_aux:
            return logits, score, aux
        return logits, score, None

    # -------------------------
    # Parameter groups helper
    # -------------------------
    def get_param_groups(self, base_lr: float = 1e-4, backbone_lr_mult: float = 0.25, weight_decay: float = 1e-5) -> List[Dict]:
        """
        Return optimizer param groups with deduplication and a backbone multiplier.
        Useful call: optim.AdamW(model.get_param_groups(...))
        """
        # collect all params unique by id
        seen = set()
        backbone_params = []
        other_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            # heuristic: parameters that belong to backbone module get backbone multiplier
            if name.startswith("backbone"):
                backbone_params.append(p)
            else:
                other_params.append(p)

        groups = [
            {"params": backbone_params, "lr": base_lr * backbone_lr_mult, "weight_decay": weight_decay},
            {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
        ]
        return groups


# -------------------------
# Smoke test to confirm wiring
# -------------------------
def smoke_test(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    print("Running MambaAlign smoke test on device:", device)
    model = MambaAlign(x_in_ch=1, pretrained_backbone=False).to(device)
    B = 2
    H = 512
    W = 512
    rgb = torch.randn(B, 3, H, W, device=device)
    x = torch.randn(B, 1, H, W, device=device)
    with torch.no_grad():
        logits, score, aux = model(rgb, x, return_aux=True)
    print("logits shape:", logits.shape)
    print("score shape:", score.shape)
    print("aux keys:", list(aux.keys()))
    assert logits.shape[0] == B
    assert logits.shape[2] == H and logits.shape[3] == W
    assert score.shape[0] == B
    print("MambaAlign smoke test passed.")


if __name__ == "__main__":
    smoke_test()
